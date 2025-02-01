from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from datetime import datetime


class PortfolioOptimizer:
    def __init__(self, user_risk_profile):
        self.memory = ConversationBufferMemory(memory_key="portfolio_history")
        self.llm = OpenAI(model_name="text-davinci-003")
        self.user_risk_profile = user_risk_profile
        self.setup_tools()

    def setup_tools(self):
        self.tools = [
            Tool(
                name="Sentiment Analysis",
                func=self.perform_sentiment_analysis,
                description="Analyzes sentiment for a given asset based on recent news and social media data."
            ),
            Tool(
                name="Risk Assessment",
                func=self.assess_risk,
                description="Evaluates the risk of an asset based on volatility and user risk tolerance."
            ),
            Tool(
                name="Portfolio Rebalancing",
                func=self.rebalance_portfolio,
                description="Optimizes portfolio allocation based on sentiment, risk, and performance."
            ),
            Tool(
                name="Recommendation Generator",
                func=self.generate_recommendation,
                description="Provides actionable recommendations for portfolio adjustments."
            )
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory
        )

    def perform_sentiment_analysis(self, asset):
        prompt = PromptTemplate(
            input_variables=["asset"],
            template=("Analyze the current sentiment for the asset '{asset}' based on recent news and social media. "
                      "Rate the sentiment from -1 (very negative) to +1 (very positive).")
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        sentiment_score = chain.run(asset=asset)
        return sentiment_score

    def assess_risk(self, asset, sentiment_score, market_volatility):
        prompt = PromptTemplate(
            input_variables=["asset", "sentiment_score", "market_volatility", "user_risk_profile"],
            template=("For the asset '{asset}', with a sentiment score of {sentiment_score} and market volatility "
                      "of {market_volatility}, assess the risk based on a '{user_risk_profile}' risk profile.")
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        risk_assessment = chain.run(
            asset=asset,
            sentiment_score=sentiment_score,
            market_volatility=market_volatility,
            user_risk_profile=self.user_risk_profile
        )
        return risk_assessment

    def rebalance_portfolio(self, asset, risk_assessment):
        prompt = PromptTemplate(
            input_variables=["asset", "risk_assessment"],
            template=("Based on the risk assessment '{risk_assessment}' for the asset '{asset}', "
                      "suggest an optimal portfolio rebalancing strategy.")
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        rebalancing_strategy = chain.run(asset=asset, risk_assessment=risk_assessment)
        return rebalancing_strategy

    def generate_recommendation(self, asset, rebalancing_strategy):
        prompt = PromptTemplate(
            input_variables=["asset", "rebalancing_strategy"],
            template=("Based on the portfolio rebalancing strategy '{rebalancing_strategy}' for the asset '{asset}', "
                      "generate actionable recommendations for the user.")
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        recommendation = chain.run(asset=asset, rebalancing_strategy=rebalancing_strategy)
        return recommendation

    def optimize_portfolio(self, asset, market_volatility):
        # Step 1: Perform Sentiment Analysis
        sentiment_score = self.perform_sentiment_analysis(asset)

        # Step 2: Assess Risk
        risk_assessment = self.assess_risk(asset, sentiment_score, market_volatility)

        # Step 3: Rebalance Portfolio
        rebalancing_strategy = self.rebalance_portfolio(asset, risk_assessment)

        # Step 4: Generate Recommendation
        recommendation = self.generate_recommendation(asset, rebalancing_strategy)

        # Compile the optimization response
        response = {
            "asset": asset,
            "sentiment_score": sentiment_score,
            "risk_assessment": risk_assessment,
            "rebalancing_strategy": rebalancing_strategy,
            "recommendation": recommendation
        }
        return response


# Example Usage
if __name__ == "__main__":
    optimizer = PortfolioOptimizer(user_risk_profile="Moderate")

    # Example 1: High Negative Sentiment for Tesla
    print("Example 1 - High Negative Sentiment for Tesla:")
    response = optimizer.optimize_portfolio(asset="Tesla", market_volatility="High")
    print(response)

    # Example 2: Positive Sentiment Surge for Apple
    print("\nExample 2 - Positive Sentiment Surge for Apple:")
    response = optimizer.optimize_portfolio(asset="Apple", market_volatility="Low")
    print(response)

    # Example 3: Neutral Sentiment for Amazon with Medium Volatility
    print("\nExample 3 - Neutral Sentiment for Amazon with Medium Volatility:")
    response = optimizer.optimize_portfolio(asset="Amazon", market_volatility="Medium")
    print(response)

#https://chatgpt.com/c/6727d267-1b20-800c-93f0-4993b4935e96