from langchain.document_loaders import TextLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from scipy.optimize import minimize
import numpy as np


class AdaptiveFinancialAdvisor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(embedding_function=self.embeddings)
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.llm = OpenAI(model_name="text-davinci-003")

        self.setup_tools()

    def setup_tools(self):
        self.tools = [
            Tool(
                name="Portfolio Optimization",
                func=self.optimize_portfolio,
                description="Optimizes portfolio based on risk tolerance and budget."
            ),
            Tool(
                name="Sentiment Analysis",
                func=self.analyze_sentiment,
                description="Analyzes sentiment from news or social media for investment insights."
            ),
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory
        )

    def optimize_portfolio(self, query):
        prompt = PromptTemplate(
            input_variables=["budget", "risk_level"],
            template=(
                "Given a budget of {budget} and a risk level of {risk_level}, "
                "suggest an optimal portfolio allocation considering tech stocks, "
                "bonds, healthcare stocks, and commodities."
            )
        )
        # Dummy allocation based on a simplistic optimization for illustration
        allocation = self._get_allocation(float(query["budget"]), query["risk_level"])
        return allocation

    def _get_allocation(self, budget, risk_level):
        risk_weights = {"low": 0.2, "medium": 0.5, "high": 0.8}
        weights = np.array([risk_weights.get(risk_level, 0.5), 0.3, 0.2, 0.1])
        allocation = weights * budget
        return {
            "tech_stocks": allocation[0],
            "bonds": allocation[1],
            "healthcare_stocks": allocation[2],
            "commodities": allocation[3]
        }

    def analyze_sentiment(self, news_headline):
        prompt = PromptTemplate(
            input_variables=["news"],
            template="Analyze the sentiment of the following news headline for financial impact: {news}"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        sentiment_analysis = chain.run(news=news_headline)
        impact = self._determine_impact(sentiment_analysis)
        return impact

    def _determine_impact(self, sentiment):
        if "negative" in sentiment.lower():
            impact = "Consider reducing tech stock exposure and increasing bonds."
        elif "positive" in sentiment.lower():
            impact = "Consider increasing tech stock exposure."
        else:
            impact = "Neutral sentiment; maintain current allocations."
        return impact

    def user_query(self, query_type, query_data):
        if query_type == "optimize_portfolio":
            response = self.optimize_portfolio(query_data)
        elif query_type == "analyze_sentiment":
            response = self.analyze_sentiment(query_data["news_headline"])
        else:
            response = "Unknown query type."
        return response


# Example Usage
if __name__ == "__main__":
    advisor = AdaptiveFinancialAdvisor()

    # Portfolio optimization example
    query_data = {"budget": 50000, "risk_level": "medium"}
    print("Portfolio Optimization:")
    print(advisor.user_query("optimize_portfolio", query_data))

    # Sentiment analysis example with news
    print("\nSentiment Analysis for News Headline:")
    news_data = {"news_headline": "Federal Reserve Hints at Possible Rate Hike Next Quarter."}
    print(advisor.user_query("analyze_sentiment", news_data))

    # Sentiment analysis example with social media sentiment
    print("\nSentiment Analysis for Social Media:")
    social_media_data = {"news_headline": "Crypto investments are highly risky and unsustainable, says analyst."}
    print(advisor.user_query("analyze_sentiment", social_media_data))

# https://chatgpt.com/c/6727d267-1b20-800c-93f0-4993b4935e96
