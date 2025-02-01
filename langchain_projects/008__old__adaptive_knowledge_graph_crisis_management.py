from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.utilities import Neo4jGraph  # Example for knowledge graph interface
from langchain.callbacks import LoggingCallbackHandler
from langchain.schema import Entity, Relationship


class CrisisManagementExpertSystem:
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="crisis_context")
        self.llm = OpenAI(model_name="gpt-4")
        self.graph = Neo4jGraph()  # Assuming Neo4j as the backend for knowledge graph
        self.setup_tools()

    def setup_tools(self):
        self.tools = [
            Tool(
                name="Dynamic Knowledge Graph Creation",
                func=self.create_knowledge_graph,
                description="Automatically builds a knowledge graph based on detected entities and relationships."
            ),
            Tool(
                name="Inference and Reasoning",
                func=self.reason_over_graph,
                description="Uses multi-step reasoning over the knowledge graph to recommend actions."
            ),
            Tool(
                name="Real-Time Crisis Adaptability",
                func=self.update_graph_with_new_data,
                description="Updates the knowledge graph with new information to adapt recommendations."
            ),
            Tool(
                name="Explainability Module",
                func=self.generate_explanation,
                description="Generates justifications for each recommendation using insights from the knowledge graph."
            )
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            callback_manager=LoggingCallbackHandler()
        )

    def create_knowledge_graph(self, crisis_event):
        # Example of parsing the event to extract entities and relationships
        entities = self.extract_entities(crisis_event)
        relationships = self.determine_relationships(entities)
        for entity in entities:
            self.graph.add_entity(Entity(name=entity))
        for rel in relationships:
            self.graph.add_relationship(Relationship(*rel))
        return "Knowledge graph created for the event."

    def extract_entities(self, text):
        # Dummy function to simulate entity extraction
        return ["Forest Fire", "Residential Area", "Winds"]

    def determine_relationships(self, entities):
        # Dummy function to simulate relationship extraction
        return [("Forest Fire", "Proximity", "Residential Area"), ("Forest Fire", "Affected by", "Winds")]

    def reason_over_graph(self, context):
        prompt = PromptTemplate(
            input_variables=["context"],
            template=("Based on the current crisis context in the knowledge graph, provide a step-by-step "
                      "reasoning process to recommend the most effective actions.")
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        recommendation = chain.run(context=context)
        return recommendation

    def update_graph_with_new_data(self, new_info):
        entities = self.extract_entities(new_info)
        relationships = self.determine_relationships(entities)
        for entity in entities:
            self.graph.add_entity(Entity(name=entity))
        for rel in relationships:
            self.graph.add_relationship(Relationship(*rel))
        return "Knowledge graph updated with new data."

    def generate_explanation(self, recommendation):
        prompt = PromptTemplate(
            input_variables=["recommendation"],
            template=("Explain the reasoning behind the recommended action: '{recommendation}', "
                      "based on relationships and entities in the knowledge graph.")
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        explanation = chain.run(recommendation=recommendation)
        return explanation

    def respond(self, crisis_event):
        # Step 1: Create Knowledge Graph
        self.create_knowledge_graph(crisis_event)

        # Step 2: Generate Inference and Recommendations
        response = self.reason_over_graph(context=crisis_event)

        # Step 3: Explain Recommendations
        explanation = self.generate_explanation(recommendation=response)

        return response, explanation


# Example Usage
if __name__ == "__main__":
    agent = CrisisManagementExpertSystem()

    # Example 1: Forest Fire Scenario
    print("Example 1 - Forest Fire Scenario:")
    response, explanation = agent.respond("Forest fire detected near residential area with strong winds forecasted.")
    print(f"Recommended Action: {response}\nExplanation: {explanation}\n")

    # Example 2: Cyberattack on Government Databases
    print("Example 2 - Cyberattack on Government Databases:")
    response, explanation = agent.respond(
        "Large-scale cyberattack detected on government databases. Sensitive citizen data at risk.")
    print(f"Recommended Action: {response}\nExplanation: {explanation}\n")

    # Example 3: River Flooding Scenario
    print("Example 3 - River Flooding Scenario:")
    response, explanation = agent.respond(
        "Heavy rains leading to rapid river flooding in urban area. Residents report flooding in low-lying zones.")
    print(f"Recommended Action: {response}\nExplanation: {explanation}\n")

#https://chatgpt.com/c/6727d267-1b20-800c-93f0-4993b4935e96