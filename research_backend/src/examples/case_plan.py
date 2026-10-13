from app.nodes.plan_decompose import plan_decompose
from app.state import State
from rich import print as rprint

state: State = State(
    user_query="调研欧美国家 酒店行业 在AI Agent 的落地使用"
)
result = plan_decompose(state)
rprint(result)

from app.nodes.research_run import research_run

config = {
    "recursion_limit":2
}
result = research_run(result,config=config)
rprint(result)
