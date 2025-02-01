from langchain.document_loaders import TextLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool


class BusinessStrategySimulator:
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="strategy_history")
        self.llm = OpenAI(model_name="text-davinci-003")
        self.setup_tools()

    def setup_tools(self):
        self.tools = [
            Tool(
                name="Scenario Analysis",
                func=self.analyze_scenario,
                description="Analyzes a business scenario and its impact on KPIs."
            ),
            Tool(
                name="KPI Adjustment",
                func=self.adjust_kpis,
                description="Adjusts KPIs based on the scenario impact."
            ),
            Tool(
                name="Strategic Recommendation Generator",
                func=self.generate_recommendations,
                description="Generates strategic recommendations based on scenario analysis."
            ),
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory
        )

    def analyze_scenario(self, scenario):
        prompt = PromptTemplate(
            input_variables=["scenario"],
            template=(
                "Analyze the following business scenario: '{scenario}'. Identify potential impacts on "
                "key performance indicators such as revenue, market share, and operational costs."
            )
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        analysis = chain.run(scenario=scenario)
        return analysis

    def adjust_kpis(self, analysis):
        prompt = PromptTemplate(
            input_variables=["analysis"],
            template=(
                "Based on the following analysis, '{analysis}', adjust key performance indicators (KPIs) "
                "such as revenue, market share, and customer satisfaction."
            )
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        kpi_adjustments = chain.run(analysis=analysis)
        return kpi_adjustments

    def generate_recommendations(self, scenario):
        prompt = PromptTemplate(
            input_variables=["scenario"],
            template=(
                "Based on the following business scenario, '{scenario}', provide strategic recommendations "
                "to improve outcomes. Consider adjustments in pricing, production, marketing, or customer service."
            )
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        recommendations = chain.run(scenario=scenario)
        return recommendations

    def simulate_strategy(self, scenario):
        # Step 1: Scenario Analysis
        analysis = self.analyze_scenario(scenario)

        # Step 2: Adjust KPIs based on Scenario Analysis
        kpi_adjustments = self.adjust_kpis(analysis)

        # Step 3: Generate Strategic Recommendations
        recommendations = self.generate_recommendations(scenario)

        # Compile the simulation response
        response = {
            "scenario": scenario,
            "analysis": analysis,
            "kpi_adjustments": kpi_adjustments,
            "recommendations": recommendations
        }
        return response


# Example Usage
if __name__ == "__main__":
    simulator = BusinessStrategySimulator()

    # Example 1: Increase in Customer Demand due to Seasonality
    print("Example 1 - Customer Demand Increase:")
    response = simulator.simulate_strategy("Increase in customer demand due to seasonality.")
    print(response)

    # Example 2: Competitor Reduces Prices
    print("\nExample 2 - Competitor Price Reduction:")
    response = simulator.simulate_strategy("Competitor reduces prices by 10%.")
    print(response)

    # Example 3: Supply Chain Disruption
    print("\nExample 3 - Supply Chain Disruption:")
    response = simulator.simulate_strategy("Supply chain disruption due to unexpected delays.")
    print(response)

#https://chatgpt.com/c/6727d267-1b20-800c-93f0-4993b4935e96