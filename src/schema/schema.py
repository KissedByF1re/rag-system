from typing import Any, Literal, NotRequired

from pydantic import BaseModel, Field, SerializeAsAny
from typing_extensions import TypedDict

from schema.models import AllModelEnum, OpenAIModelName


class AgentInfo(BaseModel):
    """Info about an available agent."""

    key: str = Field(
        description="Agent key.",
        examples=["research-assistant"],
    )
    description: str = Field(
        description="Description of the agent.",
        examples=["A research assistant for generating research papers."],
    )


class ServiceMetadata(BaseModel):
    """Metadata about the service including available agents and models."""

    agents: list[AgentInfo] = Field(
        description="List of available agents.",
    )
    models: list[AllModelEnum] = Field(
        description="List of available LLMs.",
    )
    default_agent: str = Field(
        description="Default agent used when none is specified.",
        examples=["research-assistant"],
    )
    default_model: AllModelEnum = Field(
        description="Default model used when none is specified.",
    )


class UserInput(BaseModel):
    """Basic user input for the agent."""

    message: str = Field(
        description="User input to the agent.",
        examples=["What is the weather in Tokyo?"],
    )
    model: SerializeAsAny[AllModelEnum] | None = Field(
        title="Model",
        description="LLM Model to use for the agent.",
        default=OpenAIModelName.GPT_4O,
        examples=[OpenAIModelName.GPT_4O],
    )
    thread_id: str | None = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    agent_config: dict[str, Any] = Field(
        description="Additional configuration to pass through to the agent",
        default={},
        examples=[{"spicy_level": 0.8}],
    )


class StreamInput(UserInput):
    """User input for streaming the agent's response."""

    stream_tokens: bool = Field(
        description="Whether to stream LLM tokens to the client.",
        default=True,
    )


class ToolCall(TypedDict):
    """Represents a request to call a tool."""

    name: str
    """The name of the tool to be called."""
    args: dict[str, Any]
    """The arguments to the tool call."""
    id: str | None
    """An identifier associated with the tool call."""
    type: NotRequired[Literal["tool_call"]]


class ChatMessage(BaseModel):
    """Message in a chat."""

    type: Literal["human", "ai", "tool", "custom"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool", "custom"],
    )
    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: list[ToolCall] = Field(
        description="Tool calls in the message.",
        default=[],
    )
    tool_call_id: str | None = Field(
        description="Tool call that this message is responding to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: str | None = Field(
        description="Run ID of the message.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    response_metadata: dict[str, Any] = Field(
        description="Response metadata. For example: response headers, logprobs, token counts.",
        default={},
    )
    custom_data: dict[str, Any] = Field(
        description="Custom message data.",
        default={},
    )

    def pretty_repr(self) -> str:
        """Get a pretty representation of the message."""
        base_title = self.type.title() + " Message"
        padded = " " + base_title + " "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        print(self.pretty_repr())  # noqa: T201


class Feedback(BaseModel):
    """Feedback for a run, to record to LangSmith."""

    run_id: str = Field(
        description="Run ID to record feedback for.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    key: str = Field(
        description="Feedback key.",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="Feedback score.",
        examples=[0.8],
    )
    kwargs: dict[str, Any] = Field(
        description="Additional feedback kwargs, passed to LangSmith.",
        default={},
        examples=[{"comment": "In-line human feedback"}],
    )


class FeedbackResponse(BaseModel):
    status: Literal["success"] = "success"


class ChatHistoryInput(BaseModel):
    """Input for retrieving chat history."""

    thread_id: str = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )


class ChatHistory(BaseModel):
    messages: list[ChatMessage]


class ProductVariant(BaseModel):
    """Represents a product variant."""
    
    id: str = Field(
        description="Unique identifier for the variant.",
        examples=["variant123"],
    )
    volume: str | None = Field(
        description="Volume or size information for the variant.",
        examples=["100ml"],
        default=None,
    )
    price: str | None = Field(
        description="Price of the variant.",
        examples=["$25.99"],
        default=None,
    )
    availability: str | None = Field(
        description="Availability status of the variant.",
        examples=["In stock"],
        default=None,
    )
    barcode: str | None = Field(
        description="Barcode of the variant.",
        examples=["123456789"],
        default=None,
    )
    is_default: bool = Field(
        description="Whether this is the default variant for the product.",
        default=False,
    )


class ProductDetail(BaseModel):
    """Detailed product information with variants."""
    
    id: str = Field(
        description="Unique identifier for the product.",
        examples=["prod123"],
    )
    vendor_code: str | None = Field(
        description="Vendor code for the product.",
        examples=["VC12345"],
        default=None,
    )
    title: str = Field(
        description="Product title.",
        examples=["Face Cream"],
    )
    short_description: str | None = Field(
        description="Short product description.",
        default=None,
    )
    description: str | None = Field(
        description="Full product description.",
        default=None,
    )
    brand: str | None = Field(
        description="Product brand.",
        examples=["Brand X"],
        default=None,
    )
    image_link: str | None = Field(
        description="Link to product image.",
        default=None,
    )
    link: str | None = Field(
        description="Link to product page.",
        default=None,
    )
    barcode: str | None = Field(
        description="Base product barcode.",
        default=None,
    )
    is_new: bool | None = Field(
        description="Whether this is a new product.",
        default=None,
    )
    is_hit: bool | None = Field(
        description="Whether this is a hit/popular product.",
        default=None,
    )
    only_for_cosmetologist: bool | None = Field(
        description="Whether this product is only for professional cosmetologists.",
        default=None,
    )
    manufacturer_country: str | None = Field(
        description="Product manufacturing country.",
        default=None,
    )
    categories: list[str] | None = Field(
        description="Product categories.",
        default=[],
    )
    indications: list[str] | None = Field(
        description="Product indications.",
        default=[],
    )
    skin_types: list[str] | None = Field(
        description="Compatible skin types.",
        default=[],
    )
    actions: list[str] | None = Field(
        description="Product actions/effects.",
        default=[],
    )
    ingredients: list[str] | None = Field(
        description="Product ingredients.",
        default=[],
    )
    procedure_types: list[str] | None = Field(
        description="Compatible procedure types.",
        default=[],
    )
    age_groups: list[str] | None = Field(
        description="Target age groups.",
        default=[],
    )
    application_areas: list[str] | None = Field(
        description="Areas of application.",
        default=[],
    )
    indications_text: str | None = Field(
        description="Detailed indications text.",
        default=None,
    )
    composition: str | None = Field(
        description="Detailed composition text.",
        default=None,
    )
    application_area_text: str | None = Field(
        description="Detailed application area text.",
        default=None,
    )
    efficiency_and_benefits: str | None = Field(
        description="Product efficiency and benefits text.",
        default=None,
    )
    contraindications: str | None = Field(
        description="Product contraindications text.",
        default=None,
    )
    storage: str | None = Field(
        description="Product storage instructions.",
        default=None,
    )
    keywords: list[str] | None = Field(
        description="Product keywords.",
        default=[],
    )
    variants: list[ProductVariant] = Field(
        description="All product variants.",
        default=[],
    )
    requested_variant: ProductVariant = Field(
        description="The specific variant that was requested.",
    )
