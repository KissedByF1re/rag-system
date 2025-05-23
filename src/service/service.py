import json
import logging
import os
import time
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from openinference.instrumentation.langchain import LangChainInstrumentor

from agents import DEFAULT_AGENT, get_agent, get_all_agent_info
from core import settings
from memory import initialize_database
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)

import httpx

# Set up logger first
warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)

# This is a hot reload test comment - this should trigger the service to reload

# Default values for Phoenix
PHOENIX_ENABLED = False
phoenix_client = None

# Add Phoenix imports - with simple error handling
try:
    # Check if Phoenix is enabled via environment variable
    PHOENIX_ENABLED = os.getenv("PHOENIX_ENABLED", "false").lower() == "true"
    
    if PHOENIX_ENABLED:
        logger.info("Phoenix support is enabled, initializing...")
        
        # Import Phoenix client
        from phoenix.client import Client as PhoenixClient
        
        # Get Phoenix configuration from environment variables
        phoenix_host = os.getenv("PHOENIX_HOST", "bread-phoenix")
        phoenix_port = os.getenv("PHOENIX_PORT", "6006")
        phoenix_url = f"http://{phoenix_host}:{phoenix_port}"
        
        logger.info(f"Connecting to Phoenix at: {phoenix_url}")
        
        # Set environment variables for Phoenix connection
        os.environ["PHOENIX_HOST"] = phoenix_host
        os.environ["PHOENIX_PORT"] = phoenix_port
        
        # Create Phoenix client with retry
        for attempt in range(3):
            try:
                logger.info(f"Initializing Phoenix client attempt {attempt+1}/3")
                # Create client without URL parameter - it reads from environment variables
                phoenix_client = PhoenixClient()
                
                # Inspect the Phoenix client to see what attributes and methods are available
                client_attrs = [attr for attr in dir(phoenix_client) if not attr.startswith('_')]
                logger.info(f"Phoenix client attributes: {client_attrs}")
                
                # The current Phoenix client has spans, projects, and prompts modules
                # Check if we have access to the spans API for tracing
                if hasattr(phoenix_client, "spans"):
                    logger.info("Phoenix client has spans module, using it for tracing")
                    
                    # Inspect available spans methods
                    span_methods = [attr for attr in dir(phoenix_client.spans) if not attr.startswith('_')]
                    logger.info(f"Phoenix spans methods: {span_methods}")
                    
                    # Try to initialize OpenTelemetry if the spans module is available
                    break
                
                # Check if we have the projects API for organizing data
                elif hasattr(phoenix_client, "projects"):
                    logger.info("Phoenix client has projects module, using it for organization")
                    
                    # Inspect available project methods
                    project_methods = [attr for attr in dir(phoenix_client.projects) if not attr.startswith('_')]
                    logger.info(f"Phoenix projects methods: {project_methods}")
                    
                    break
                
                # Fall back to any available method
                else:
                    logger.warning("Phoenix client doesn't have expected methods, will try to use it anyway")
                    break
            except Exception as e:
                logger.warning(f"Phoenix client initialization failed on attempt {attempt+1}: {e}")
                if attempt == 2:  # Last attempt
                    raise
                time.sleep(2)  # Wait before retry
        
        # Import and set up OpenTelemetry if needed
        try:
            # Set up OpenTelemetry with Phoenix exporter
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", f"http://{phoenix_host}:4317")
            service_name = os.getenv("OTEL_SERVICE_NAME", "agent-service")
            
            # Log the endpoint for debugging
            logger.info(f"Connecting OpenTelemetry to Phoenix at: {otlp_endpoint}")
            
            resource = Resource.create({
                "service.name": service_name,
                "service.instance.id": str(uuid4()),
                "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development")
            })
            
            tracer_provider = TracerProvider(resource=resource)
            
            # Configure exporter to send spans to Phoenix
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                timeout=10  # 10 second timeout
            )
            
            # Use BatchSpanProcessor with optimized settings
            span_processor = BatchSpanProcessor(
                otlp_exporter,
                max_queue_size=1000,
                max_export_batch_size=100,
                schedule_delay_millis=1000,  # 1 second
                export_timeout_millis=10000,  # 10 seconds
            )
            tracer_provider.add_span_processor(span_processor)
            
            # Set the tracer provider
            trace.set_tracer_provider(tracer_provider)
            
            # Initialize Phoenix instrumentation for LangChain
            LangChainInstrumentor().instrument()
            logger.info("OpenTelemetry instrumentation initialized successfully")
        except Exception as otel_error:
            logger.error(f"Failed to initialize OpenTelemetry: {otel_error}")
            logger.error("OpenTelemetry tracing will not be available")
    else:
        logger.info("Phoenix support is disabled via environment variable")
except Exception as e:
    logger.error(f"Phoenix initialization failed, feedback will use LangSmith if available: {e}")
    PHOENIX_ENABLED = False
    phoenix_client = None


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Configurable lifespan that initializes the appropriate database checkpointer based on settings.
    """
    try:
        async with initialize_database() as saver:
            await saver.setup()
            agents = get_all_agent_info()
            for a in agents:
                agent = get_agent(a.key)
                agent.checkpointer = saver
            
            # Configure Phoenix retention policy after service initialization
            if PHOENIX_ENABLED:
                try:
                    await configure_phoenix_retention()
                except Exception as e:
                    logger.warning(f"Failed to configure Phoenix retention on startup: {e}")
            
            yield
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise


async def configure_phoenix_retention():
    """Configure Phoenix retention policy via GraphQL API."""
    if not PHOENIX_ENABLED or not phoenix_client:
        return
    
    try:
        phoenix_host = os.getenv("PHOENIX_HOST", "bread-phoenix")
        phoenix_port = os.getenv("PHOENIX_PORT", "6006")
        phoenix_url = f"http://{phoenix_host}:{phoenix_port}"
        
        logger.info("🔧 Phoenix Retention Policy Configuration")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Check if Phoenix is accessible
            try:
                health_response = await client.get(f"{phoenix_url}/health")
                if health_response.status_code == 200:
                    logger.info(f"✅ Phoenix is running and accessible at {phoenix_url}")
                else:
                    logger.warning(f"⚠️ Phoenix health check failed: {health_response.status_code}")
                    return
            except Exception as e:
                logger.warning(f"⚠️ Phoenix not accessible: {e}")
                return
            
            # Get current projects and their retention policies
            try:
                projects_query = {
                    "query": """
                    query {
                        projects {
                            edges {
                                node {
                                    id
                                    name
                                    traceRetentionPolicy {
                                        id
                                        name
                                        rule {
                                            ... on TraceRetentionRuleMaxDays {
                                                maxDays
                                            }
                                            ... on TraceRetentionRuleMaxCount {
                                                maxCount
                                            }
                                            ... on TraceRetentionRuleMaxDaysOrCount {
                                                maxDays
                                                maxCount
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    """
                }
                
                projects_response = await client.post(
                    f"{phoenix_url}/graphql",
                    json=projects_query,
                    headers={"Content-Type": "application/json"}
                )
                
                if projects_response.status_code == 200:
                    projects_data = projects_response.json()
                    if "data" in projects_data and projects_data["data"]["projects"]:
                        projects = projects_data["data"]["projects"]["edges"]
                        logger.info(f"📊 Found {len(projects)} Phoenix project(s)")
                        
                        for project_edge in projects:
                            project = project_edge["node"]
                            retention_policy = project["traceRetentionPolicy"]
                            current_rule = retention_policy["rule"]
                            
                            logger.info(f"   • Project: {project['name']} (ID: {project['id']})")
                            logger.info(f"     Retention Policy: {retention_policy['name']} (ID: {retention_policy['id']})")
                            
                            # Check current retention rule
                            if "maxCount" in current_rule:
                                if current_rule.get("maxCount") == 10000:
                                    logger.info(f"     ✅ Already configured for 10k traces (current: {current_rule['maxCount']})")
                                    continue
                                else:
                                    logger.info(f"     Current: {current_rule['maxCount']} traces")
                            elif "maxDays" in current_rule:
                                logger.info(f"     Current: {current_rule['maxDays']} days retention")
                            else:
                                logger.info(f"     Current rule: {current_rule}")
                            
                            logger.info(f"🚀 Updating retention policy to 10k traces for project: {project['name']}")
                            
                            # Update retention policy using the working GraphQL mutation
                            update_policy_mutation = {
                                "query": """
                                mutation UpdateRetentionPolicy($id: ID!, $rule: ProjectTraceRetentionRuleInput!) {
                                    patchProjectTraceRetentionPolicy(input: {
                                        id: $id,
                                        rule: $rule
                                    }) {
                                        node {
                                            id
                                            name
                                            rule {
                                                ... on TraceRetentionRuleMaxDaysOrCount {
                                                    maxDays
                                                    maxCount
                                                }
                                            }
                                        }
                                    }
                                }
                                """,
                                "variables": {
                                    "id": retention_policy["id"],
                                    "rule": {
                                        "maxDaysOrCount": {
                                            "maxDays": 0.0,
                                            "maxCount": 10000
                                        }
                                    }
                                }
                            }
                            
                            mutation_response = await client.post(
                                f"{phoenix_url}/graphql",
                                json=update_policy_mutation,
                                headers={"Content-Type": "application/json"}
                            )
                            
                            if mutation_response.status_code == 200:
                                mutation_data = mutation_response.json()
                                
                                if "errors" in mutation_data and mutation_data["errors"]:
                                    error_msg = mutation_data["errors"][0].get("message", "Unknown error")
                                    logger.warning(f"⚠️ GraphQL error updating retention policy: {error_msg}")
                                elif "data" in mutation_data and mutation_data["data"]:
                                    policy_data = mutation_data["data"]["patchProjectTraceRetentionPolicy"]["node"]
                                    rule_data = policy_data["rule"]
                                    logger.info(f"✅ Successfully updated retention policy:")
                                    logger.info(f"   • Policy: {policy_data['name']} (ID: {policy_data['id']})")
                                    logger.info(f"   • Max Traces: {rule_data['maxCount']}")
                                    logger.info(f"   • Max Days: {rule_data['maxDays']} (unlimited)")
                                    logger.info(f"   🎯 Phoenix will now retain up to 10,000 traces!")
                                else:
                                    logger.warning(f"⚠️ Unexpected GraphQL response: {mutation_data}")
                            else:
                                logger.warning(f"⚠️ GraphQL mutation failed: {mutation_response.status_code}")
                                try:
                                    error_detail = mutation_response.text
                                    logger.warning(f"Error detail: {error_detail}")
                                except:
                                    pass
                        
                        logger.info("🎯 Phoenix retention configuration completed")
                        
                    else:
                        logger.warning("⚠️ No projects found in Phoenix")
                        
                else:
                    logger.warning(f"⚠️ Failed to query Phoenix projects: {projects_response.status_code}")
                
            except Exception as api_error:
                logger.warning(f"ℹ️ Phoenix GraphQL API error: {api_error}")
                logger.info("💡 Manual Configuration Alternative:")
                logger.info(f"   1. Visit Phoenix UI: {phoenix_url}")
                logger.info("   2. Go to Settings → Data Retention")
                logger.info("   3. Update policy: Max Traces=10000, Max Days=0")
                
    except Exception as e:
        logger.warning(f"ℹ️ Phoenix retention configuration error: {e}")
        logger.info("💡 For manual configuration:")
        logger.info("   • Open Phoenix UI at http://localhost:8881")
        logger.info("   • Navigate to Settings → Data Retention") 
        logger.info("   • Update Default policy: Max Traces=10000, Max Days=0")


app = FastAPI(lifespan=lifespan)
router = APIRouter(dependencies=[Depends(verify_bearer)])


@router.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


async def _handle_input(
    user_input: UserInput, agent: CompiledStateGraph
) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())

    configurable = {"thread_id": thread_id, "model": user_input.model}

    if user_input.agent_config:
        if overlap := configurable.keys() & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    config = RunnableConfig(
        configurable=configurable,
        run_id=run_id,
    )

    # Check for interrupts that need to be resumed
    state = await agent.aget_state(config=config)
    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]

    if interrupted_tasks:
        # assume user input is response to resume agent execution from interrupt
        input = Command(resume=user_input.message)
    else:
        input = {"messages": [HumanMessage(content=user_input.message)]}

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id


@router.post("/{agent_id}/invoke")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    # NOTE: Currently this only returns the last message or interrupt.
    # In the case of an agent outputting multiple AIMessages (such as the background step
    # in interrupt-agent, or a tool step in research-assistant), it's omitted. Arguably,
    # you'd want to include it. You could update the API to return a list of ChatMessages
    # in that case.
    agent: CompiledStateGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)
    try:
        response_events = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])
        response_type, response = response_events[-1]
        if response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        
        # The frontend will pass this back in feedback.kwargs.span_id
        if PHOENIX_ENABLED:
            output.response_metadata["phoenix_span_id"] = str(run_id)
            
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: CompiledStateGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)

    try:
        # Process streamed events from the graph and yield messages over the SSE stream.
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "messages", "custom"]
        ):
            if not isinstance(stream_event, tuple):
                continue
            stream_mode, event = stream_event
            new_messages = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    # A simple approach to handle agent interrupts.
                    # In a more sophisticated implementation, we could add
                    # some structured ChatMessage type to return the interrupt value.
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue
                    updates = updates or {}
                    update_messages = updates.get("messages", [])
                    # special cases for using langgraph-supervisor library
                    if node == "supervisor":
                        # Get only the last AIMessage since supervisor includes all previous messages
                        ai_messages = [msg for msg in update_messages if isinstance(msg, AIMessage)]
                        if ai_messages:
                            update_messages = [ai_messages[-1]]
                    if node in ("research_expert", "math_expert", "opensearch_expert"):
                        # By default the sub-agent output is returned as an AIMessage.
                        # Convert it to a ToolMessage so it displays in the UI as a tool response.
                        msg = ToolMessage(
                            content=update_messages[0].content,
                            name=node,
                            tool_call_id="",
                        )
                        update_messages = [msg]
                    new_messages.extend(update_messages)

            if stream_mode == "custom":
                new_messages = [event]

            for message in new_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                    
                    # For Phoenix feedback, we use the same ID as the run_id
                    # The frontend will pass this back in feedback.kwargs.span_id
                    if PHOENIX_ENABLED:
                        chat_message.response_metadata["phoenix_span_id"] = str(run_id)
                        
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                # Drop them.
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # Empty content in the context of OpenAI usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content.
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
    except Exception as e:
        logger.error(f"Error in message generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


def _sse_response_example() -> dict[int, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run directly to Phoenix.
    """
    if PHOENIX_ENABLED and phoenix_client:
        # Get the Phoenix span ID from kwargs if it exists, otherwise use run_id
        span_id = feedback.kwargs.get("phoenix_span_id", feedback.run_id)
        
        logger.info(f"Recording feedback for span_id: {span_id} (score: {feedback.score})")
        
        # Use the Phoenix direct logging approach
        try:
            # Create direct API call to Phoenix
            async with httpx.AsyncClient() as client:
                annotation_payload = {"data": [{
                            "span_id": span_id,
                            "name": "user feedback",
                            "annotator_kind": "HUMAN",
                            "result": {"label": feedback.key, "score": feedback.score},
                            "metadata": {},
                        }
                    ]
                }

                await client.post(
                    f"http://{phoenix_host}:{phoenix_port}/v1/span_annotations?sync=false",
                    json=annotation_payload,
                    headers={}
                )

        except Exception as api_err:
            logger.warning(f"Failed to record feedback via API: {api_err}")

    return FeedbackResponse(success=True)


@router.post("/history")
def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    # TODO: Hard-coding DEFAULT_AGENT here is wonky
    agent: CompiledStateGraph = get_agent(DEFAULT_AGENT)
    try:
        state_snapshot = agent.get_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


app.include_router(router)
