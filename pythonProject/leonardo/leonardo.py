import asyncio
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Any
import numpy as np
import re
from nltk.tokenize import sent_tokenize

#let us measure the program
import time

from openai import AsyncOpenAI, OpenAI

load_dotenv()


# === 1. SETUP LLM (You can swap this for your own model) ===
#OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#asynchronous
async def async_call_llm(prompt: str, temperature: float = 0.0, model: str = "gpt-4o-2024-11-20") -> str:
    """Send a prompt to an LLM."""
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content

#synchronous
def call_llm(prompt: str, temperature: float = 0.0, model: str = "gpt-4o-2024-11-20") -> str:
    """Send a prompt to an LLM."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content


# === 2. KNOWLEDGE GRAPH EXTRACTION ===

async def extract_entities(text: str) -> List[str]:
    """Extract a list of entities from the text."""
    prompt = f""" You are a top-tier algorithm designed for extracting entities in structured formats.
                These entities will be used to build a knowledge graph.
                Your task is to identify the entities requested with the user prompt from a given text.
                You must generate the output in a JSON format containing a list with JSON objects.
                Each object should have the key: "entities".
                The "entities" key must contain a list of strings, where each string is an extracted entity.
                Attempt to extract as many entities as you can.
                It is very important to cover all the entities in the text!
                Entities can be in many types, such as people, organizations, locations, events, positions, dates, roles, titles, objects, and more.
                If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), always use the most complete identifier for that entity.
                IMPORTANT NOTES:
                - Don’t add any explanation and text.
                - It is very important to cover all the entities in the text!
                - Remember to cover dates.
                - Remember to cover events eg. publication of book, employment, graduation, marriage.
                - All numbers, dates has to be in string format. eg. "2023", "1980s".
                ================================ Human Message =================================
                Below are a number of examples of text and their extracted entities.
                [{{'text': "Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who served as the 45th president of the United States from 2017 to 2021. Having won the 2024 presidential election as the nominee of the Republican Party, he is the president-elect and will be inaugurated as the 47th president on January 20, 2025. Trump graduated with a bachelor’s degree in economics from the University of Pennsylvania in 1968.", ’entities’: [’Donald John Trump’, ’American politician’, ’media personality’, ’businessman’, ’45th president’, ’United States’, ’2017’, ’2021’, ’2024 presidential election’, ’Republican Party’, ’president-elect’, ’47th president’, ’January 20, 2025’, ’University of Pennsylvania’, ’graduation’, ’bachelor’, ’degree’, ’economics’, ’1968’]}}]
                For the following text, extract entities as in the provided example.
                Text: {text} 
    Return the list only."""
    response = await async_call_llm(prompt)
    entities = [e.strip() for e in response.split("\n") if e.strip()]
    return entities


async def extract_relations(text: str, entities: List[str]) -> List[str]:
    """Extract possible relation types between entities."""
    entities_str = ", ".join(entities)
    prompt = f"""You are a top-tier algorithm designed for extracting relationship types in structured formats.
        These relationship types will be used to build a knowledge graph in the future.
        Your task is to identify the relationship types requested with the user prompt from a given text and list of entities.
        You must generate the output in a JSON format containing a list with JSON objects.
        Each object should have the key: "relationship_types".
        The "relationship_types" key must contain a list of strings, where each string is an extracted relationship type.
        Attempt to extract as many relationship types as you can.
        It is very important to cover all the relationship types in the text!
        Ensure the relationships accurately describe the interaction or connection between entities.
        Relationship types should be simple and concise, but also specific and informative, to cover all the relationships between entities in the text.
        If it is possible, use the WikiData relationship types. If not, use the most appropriate relationship type.
        IMPORTANT NOTES:
        - Don’t add any explanation and text.
        - It is very important to cover all the relationship types in the text!
        - Omit long and complex relationship types.
        - Use double quotes, not single quotes for json format.
        ================================ Human Message =================================
        Below are a number of examples of text and their extracted relationship types.
        [{{'text': "Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who served as the 45th president of the United States from 2017 to 2021. Having won the 2024 presidential election as the nominee of the Republican Party, he is the president-elect and will be inaugurated as the 47th president on January 20, 2025. Trump graduated with a bachelor’s degree in economics from the University of Pennsylvania in 1968.", ’entities’: [’Donald John Trump’, ’American politician’, ’media personality’, ’businessman’, ’45th president’, ’United States’, ’2017’, ’2021’, ’2024 presidential election’, ’Republican Party’, ’president-elect’, ’47th president’, ’January 20, 2025’, ’University of Pennsylvania’, ’graduation’, ’bachelor’, ’degree’, ’economics’, ’1968’], ’relationship_types’: [’DATE OF BIRTH’, ’NATIONALITY’, ’OCCUPATION’, ’POSITION HELD’, ’TERM OF OFFICE’, ’MEMBER OF’, ’ELECTED AS’, ’NOMINEE OF’, ’FUTURE POSITION’, ’DATE OF START’, ’DATE OF END’, ’EDUCATED AT’, ’DEGREE IN’, ’DEGREE FROM’, ’DEGREE TYPE’, ’DATE OF GRADUATION’]}}]
        For the following text and entities, extract relationship types as in the provided example.
        Text: {text}
        Entities: {entities_str} 
    List only the relations."""
    response = await async_call_llm(prompt)
    relations = [r.strip() for r in response.split("\n") if r.strip()]
    return relations


async def extract_facts_for_sentence(text: str, entities: List[str], relations: List[str], sentence: str):
    """Extract triples (head, relation, tail) as facts."""
    prompt = f""" You are a top-tier algorithm designed for extracting information in CSV format to build a knowledge graph.
                            In input, you will receive: sentence, full text, list of entities, and list of relationships.
                            Your task is to identify facts (triples) consisting of a subject, relation, and object from a given sentence.
                            Do not generate relationships that are defined in the full text but not in the sentence.
                            You must generate the output in CSV format with semicolon ";" as the separator, and lines separated by "\n".
                            The first line should be the header: subject;relation;object
                            Each subsequent line should contain one triple with the following columns:
                            - subject: the text of the extracted entity which is the subject of the relation
                            - relation: the type of relation between the subject and object
                            - object: the text of the extracted entity which is the object of the relation
                            The extracted entities must be from the list provided by the user.
                            The relations must be from the list of allowed relation types provided by the user.
                            Attempt to extract as many facts / triples as you can.
                            Entities can be in many types, such as people, organizations, locations, dates, roles, titles, objects, and more.
                            Maintain Entity Consistency: When extracting entities, it’s vital to ensure consistency.
                            If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), always use the most complete identifier for that entity.
                            The knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
                            The knowledge graph should be complete, so it should cover all the facts in the sentence.
                            IMPORTANT NOTES:
                            - Don’t add any explanation and text.
                            - It is very important to cover all triples (facts) in the sentence!
                            - Full text is only for understanding the context. Do not add any triples that are not in the sentence.
                            - Output must be in CSV format with semicolon (;) separators. Lines should be separated by "\n".
                            - Do not include quotes around the values unless they contain semicolons 
                             For the following sentence, full text, entities, and relation types, extract facts / triples as in the provided sentence.
                            Your response should be a list of semicolon separated values with header, eg:
                            ‘‘‘
                            subject;relation;object
                            Donald John Trump;DEGREE FROM;University of Pennsylvania
                            Donald John Trump;DEGREE IN;economics
                            ‘‘‘
                            Below are a number of examples of text and their extracted facts.
                            [{{'full_text': "Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who served as the 45th president of the United States from 2017 to 2021. Having won the 2024 presidential election as the nominee of the Republican Party, he is the president-elect and will be inaugurated as the 47th president on January 20, 2025. Trump graduated with a bachelor’s degree in economics from the University of Pennsylvania in 1968.", ’sentence’: "Trump graduated with a bachelor’s degree in economics from the University of Pennsylvania in 1968.", ’allowed entities’: [’Donald John Trump’, ’American politician’, ’media personality’, ’businessman’, ’45th president’, ’United States’, ’2017’, ’2021’, ’2024 presidential election’, ’Republican Party’, ’president-elect’, ’47th president’, ’January 20, 2025’, ’University of Pennsylvania’, ’graduation’, ’bachelor’, ’degree’, ’economics’, ’1968’], ’allowed relation types’: [’DATE OF BIRTH’, ’NATIONALITY’, ’OCCUPATION’, ’POSITION HELD’, ’TERM OF OFFICE’, ’MEMBER OF’, ’ELECTED AS’, ’NOMINEE OF’, ’FUTURE POSITION’, ’DATE OF START’, ’DATE OF END’, ’EDUCATED AT’, ’DEGREE IN’, ’DEGREE FROM’, ’DEGREE TYPE’, ’DATE OF GRADUATION’], ’triples’: ’subject;relation;object\nDonald John Trump;DEGREE FROM;University of Pennsylvania\nDonald John Trump;DEGREE IN;economics\nDonald John Trump;DEGREE TYPE;bachelor\nDonald John Trump;DATE OF GRADUATION;1968\n’}}]
                            Use the following entities, don’t use other entity that is not defined below:
                            # ALLOWED ENTITIES:
                            {entities}
                            Use the following relation types, don’t use other relation that is not defined below:
                            # ALLOWED RELATION TYPES:
                            {relations}
                            Full text: {text} 
                            Sentence: {sentence} 
            Format each fact exactly as (head, relation, tail) on a new line."""

    response = await async_call_llm(prompt)
    facts = re.findall(r'\((.*?)\)', response)
    triples = [tuple(map(str.strip, fact.split(','))) for fact in facts]
    return sentence, triples

async def extract_facts(text: str, entities: List[str], relations: List[str]) -> Dict[str, List[Tuple[str, str, str]]]:
    sentences = sent_tokenize(text)
    facts_per_sentence = {}

    tasks = [ extract_facts_for_sentence(sentence=sentence, text=text, entities=entities, relations=relations) for sentence in sentences]

    # Run all tasks in parallel
    results = await asyncio.gather(*tasks)

    # Build dictionary: sentence -> facts
    facts_per_sentence = {sentence: facts for sentence, facts in results}

    return facts_per_sentence

# === 3. SAMPLE RESPONSES GENERATION ===

async def generate_samples(prompt: str, n_samples: int = 5) -> List[str]:
    """Generate multiple stochastic samples for the same prompt."""
    tasks = [async_call_llm(prompt, temperature=0.8) for _ in range(n_samples)]

    samples = await asyncio.gather(*tasks)
    return samples



# === 4. FACT-LEVEL HALLUCINATION SCORING ===
#ℰS=ℰp∪⋃u∈U{h,t∣(h,r,t)∈K⁢Gu}
#ℛS=ℛp∪⋃u∈U{r∣(h,r,t)∈K⁢Gu}
def build_ES_RS(entities_p: List[str], relations_p: List[str], facts_per_sentence: Dict[str, List[Tuple[str, str, str]]]) -> Tuple[List[str], List[str]]:
    additional_entities = set()
    additional_relations = set()

    for facts in facts_per_sentence.values():
        for h, r, t in facts:
            additional_entities.update([h.strip(), t.strip()])
            additional_relations.add(r.strip())

    ES = list(set(entities_p) | additional_entities)
    RS = list(set(relations_p) | additional_relations)

    return ES, RS


#K⁢Gs=L⁢L⁢Msample_facts⁢(s,ℰS,ℛS)
def extract_sample_knowledge_graph(sample_text: str, ES: List[str], RS: List[str]) -> List[Tuple[str, str, str]]:
    """
    For a single sample text, extract the knowledge graph using expanded entity set (ES) and relation set (RS).
    """
    prompt = f"""
     You are a top-tier algorithm designed for extracting information in CSV format to build a knowledge graph.
    Your task is to identify facts (triples) consisting of a subject, relation, and object from a given text.
    You must generate the output in CSV format with semicolon ";" as the separator, and lines separated by "\n".
    The first line should be the header: subject;relation;object
    Each subsequent line should contain one triple with the following columns:
    - subject: the text of the extracted entity which is the subject of the relation
    - relation: the type of relation between the subject and object
    - object: the text of the extracted entity which is the object of the relation
    The extracted entities must be from the list provided by the user.
    The relations must be from the list of allowed relation types provided by the user.
    Attempt to extract as many facts / triples as you can.
    Entities can be in many types, such as people, organizations, locations, dates, roles, titles, objects, and more.
    Maintain Entity Consistency: When extracting entities, it’s vital to ensure consistency.
    If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), always use the most complete identifier for that entity.
    The knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
    The knowledge graph should be complete, so it should cover all the facts in the text.
    IMPORTANT NOTES:
    - Don’t add any explanation and text.
    - It is very important to cover all triples (facts) in the text!
    - Output must be in CSV format with semicolon (;) separators. Lines should be separated by "\n".
    - Do not include quotes around the values unless they contain semicolons
    ================================ Human Message =================================
    For the following text, allowed entities and allowed relation types, extract facts / triples as in the provided example.
    Your response should be a list of semicolon separated values with header, eg:
    ‘‘‘
    subject;relation;object
    Donald John Trump;DATE OF BIRTH;June 14, 1946
    Donald John Trump;NATIONALITY;American politician
    ‘‘‘
    Below are a number of examples of text and their extracted facts.
    [{{'text': "Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who served as the 45th president of the United States from 2017 to 2021. Having won the 2024 presidential election as the nominee of the Republican Party, he is the president-elect and will be inaugurated as the 47th president on January 20, 2025. Trump graduated with a bachelor’s degree in economics from the University of Pennsylvania in 1968.", ’allowed entities’: [’Donald John Trump’, ’American politician’, ’media personality’, ’businessman’, ’45th president’, ’United States’, ’2017’, ’2021’, ’2024 presidential election’, ’Republican Party’, ’president-elect’, ’47th president’, ’January 20, 2025’, ’University of Pennsylvania’, ’graduation’, ’bachelor’, ’degree’, ’economics’, ’1968’], ’allowed relation types’: [’DATE OF BIRTH’, ’NATIONALITY’, ’OCCUPATION’, ’POSITION HELD’, ’TERM OF OFFICE’, ’MEMBER OF’, ’ELECTED AS’, ’NOMINEE OF’, ’FUTURE POSITION’, ’DATE OF START’, ’DATE OF END’, ’EDUCATED AT’, ’DEGREE IN’, ’DEGREE FROM’, ’DEGREE TYPE’, ’DATE OF GRADUATION’], ’triples’: ’subject;relation;object\nDonald John Trump;DATE OF BIRTH;June 14, 1946\nDonald John Trump;NATIONALITY;American politician\nDonald John Trump;OCCUPATION;media personality\nDonald John Trump;OCCUPATION;businessman\nDonald John Trump;POSITION HELD;45th president\n45th president;TERM OF OFFICE;2017\n45th president;TERM OF OFFICE;2021\nDonald John Trump;ELECTED AS;2024 presidential election\n2024 presidential election;NOMINEE OF;Republican Party\nDonald John Trump;FUTURE POSITION;47th president\n47th president;DATE OF INAUGURATION;January 20, 2025\n47th president;DATE OF START;January 20, 2025\nDonald John Trump;EDUCATED AT;University of Pennsylvania\nDonald John Trump;DEGREE FROM;University of Pennsylvania\nDonald John Trump;DEGREE IN;economics\nDonald John Trump;DATE OF GRADUATION;1968\n’}}]
    Use the following entities, don’t use other entity that is not defined below:
    # ALLOWED ENTITIES:
    {ES}
    Use the following relation types, don’t use other relation that is not defined below:
    # ALLOWED RELATION TYPES:
    {RS}
    Text: {sample_text} 
    
    Return triples in the format (head, relation, tail), one per line. 
"""
    response = call_llm(prompt, temperature=0.0)
    facts = re.findall(r'\((.*?)\)', response)
    triples = [tuple(map(str.strip, fact.split(','))) for fact in facts]
    return triples


async def fact_selfcheck_KG(fact: Tuple[str, str, str], sample_graphs: List[List[Tuple[str, str, str]]]) -> float:
    verification_prompt_template = """
     You are a top-tier algorithm designed for verification whether fact is supported by the provided knowledge graph.
    Your task is to answer whether the given fact is supported by the knowledge graph.
    Return ‘yes‘ if the fact is supported, ‘no‘ otherwise.
    You can reason over the knowledge graph and the fact to answer.
    You can infer the answer from the knowledge graph, e.g. by inferring new facts from the knowledge graph.
    Do not add new facts other than by inference from the knowledge graph.
    Fact is a triple of the form ‘(subject, predicate, object)‘. Knowledge graph is a list of facts.
    You must return a single word, either ‘yes‘ or ‘no‘.
    You must not return any other text or explanation.
    ================================ Human Message =================================
    Below are a number of examples of facts, knowledge graph and their verification results.
    - Fact: (Donald Trump, president, United States)
    Knowledge graph: [(Donald Trump, president, United States), (Donald Trump, born, 1946), (Donald Trump, graduated, University of Pennsylvania)]
    Is the fact supported by the knowledge graph?: yes
    - Fact: (Joe Biden, born, 1942)
    Knowledge graph: [(Joe Biden, proffesion, politician), (Joe Biden, president, United States), (Joe Biden, graduated, University of Delaware)]
    Is the fact supported by the knowledge graph?: no
    - Fact: (Michael Jordan, played, basketball)
    Knowledge graph: [(Michael Jordan, CEO, PepsiCo), (Michael Jordan, born, 1936), (Michael Jordan, graduated, Yale University)]
    Is the fact supported by the knowledge graph?: no
    - Fact: (Kobe Bryant, played in, NBA)
    Knowledge graph: [(Kobe Bryant, born, 1978), (Kobe Bryant, played in, Los Angeles Lakers)]
    note: Los Angeles Lakers is a part of NBA
    Is the fact supported by the knowledge graph?: yes
    - Fact: (Bryant, profession, basketball player)
    Knowledge graph: [(Kobe Bryant, born, 1978), (Kobe Bryant, played in, Los Angeles Lakers)]
    note: In this context, Bryant is Kobe Bryant, and Los Angeles Lakers is a basketball team, so Kobe Bryant is a basketball player
    Is the fact supported by the knowledge graph?: yes
    ===
    For the following fact and knowledge graph, verify if the fact is supported by the knowledge graph.
    Fact: {fact}
    Knowledge graph: {knowledge_graph}
    Is the fact supported by the knowledge graph? 
"""
    results = []

    for graph in sample_graphs:
        knowledge_graph_str = str(graph)
        prompt = verification_prompt_template.format(
            fact=fact,
            knowledge_graph=knowledge_graph_str
        )

        response = (await async_call_llm(prompt, temperature=0.0)).strip().lower()

        if response == "yes":
            results.append(0)
        elif response == "no":
            results.append(1)
        else:
            continue

    if results:
        return np.mean(results)
    else:
        return 1.0

    """
    1. fhfdshfdsjlfds."""



'''async def fact_selfcheck_text(fact: Tuple[str, str, str], samples: List[str]) -> float:
    """LLM-based check if fact is supported in sample text."""
    h, r, t = fact
    prompt_template = """  You are a top-tier algorithm designed for verification whether fact is supported by the provided context.
        Your task is to answer whether the given fact is supported by the context.
        Return ‘yes‘ if the fact is supported, ‘no‘ otherwise.
        You can reason over the context and the fact to answer.
        You can infer the answer from the context, e.g. by inferring new facts from the context.
        Do not add new facts other than by inference from the context.
        Fact is a triple of the form ‘(subject, predicate, object)‘. Context is a text.
        You must return a single word, either ‘yes‘ or ‘no‘.
        You must not return any other text or explanation.
        ================================ Human Message =================================
        Below are a number of examples of facts, context and their verification results.
        - Fact: (Donald Trump, president, United States)
        Context: Donald Trump was president of the United States. Donald Trump was born in 1946 and graduated from the University of Pennsylvania.
        Is the fact supported by the context?: yes
        - Fact: (Joe Biden, born, 1942)
        Context: Joe Biden is a politician and president of the United States. He graduated from the University of Delaware.
        Is the fact supported by the context?: no
        - Fact: (Michael Jordan, played, basketball)
        Context: Michael Jordan is the CEO of PepsiCo. He was born in 1936 and graduated from Yale University.
        Is the fact supported by the context?: no
        - Fact: (Kobe Bryant, played in, NBA)
        Context: Kobe Bryant was born in 1978 and played for the Los Angeles Lakers.
        note: Los Angeles Lakers is a part of NBA
        Is the fact supported by the context?: yes
        - Fact: (Bryant, profession, basketball player)
        Context: Kobe Bryant was born in 1978 and played for the Los Angeles Lakers.
        note: In this context, Bryant is Kobe Bryant, and Los Angeles Lakers is a basketball team, so Kobe Bryant is a basketball player
        Is the fact supported by the context?: yes
        ===
        For the following fact and context, verify if the fact is supported by the context.
        Fact: {fact}
        Context: {context}
Is the fact supported by the context? """
    results = []
    for s in samples:
        prompt = prompt_template.format(fact = fact, context = s)
        answer = (await async_call_llm(prompt, temperature=0.0)).lower()
        if "yes" in answer:
            results.append(0)
        elif "no" in answer:
            results.append(1)
    if results:
        return np.mean(results)
    else:
        return 1.0  # default if no valid response'''


# === 5. AGGREGATION: SENTENCE- AND PASSAGE-LEVEL ===

"""def aggregate_fact_scores(fact_scores: Dict[Tuple[str, str, str], float], method: str = "max") -> float:
    Aggregate fact scores to a single hallucination score.
    scores = list(fact_scores.values())
    if not scores:
        return 0.0
    if method == "mean":
        return np.mean(scores)
    elif method == "max":
        return np.max(scores)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")"""


# === 6. FULL PIPELINE ===

async def process_sample(sample: str) -> Dict:
    entities = await extract_entities(sample)
    relations = await extract_relations(sample, entities)
    facts = await extract_facts(text=sample, entities=entities, relations=relations)
    return facts

async def score_fact(fact, sample_graphs):
        return fact, await fact_selfcheck_KG(fact, sample_graphs)

async def fact_selfcheck_pipeline(prompt: str, response_text: str, n_samples: int = 5, method: str = "text",
                            agg_method: str = "mean") -> Dict[str, Any]:
    """Complete FactSelfCheck pipeline."""

    start_time = time.time()
    # 1. Extract KGs
    entities = await extract_entities(response_text)
    print(f"Extracted entities, the time gone is {time.time() - start_time}")
    relations = await extract_relations(response_text, entities)
    print(f"Extracted relations, the time gone is {time.time() - start_time}")
    facts = await extract_facts(response_text, entities, relations)
    print(f"Extracted facts, the time gone is {time.time() - start_time}")

    #sample generation
    samples = await generate_samples(prompt, n_samples)
    print(f"Generated samples, the time gone is {time.time() - start_time}")

    #1. Extract entities
    tasks = [process_sample(sample) for sample in samples]
    sample_graphs = await asyncio.gather(*tasks)
    print(f"Generated sample graphs, the time gone is {time.time() - start_time}")
    # 2. Score facts
    tasks = [score_fact(fact, sample_graphs= sample_graphs) for fact in facts]
    results = await asyncio.gather(*tasks)
    print(f"Generated fact scores, the time gone is {time.time() - start_time}")

    fact_scores = {fact: float(score) for fact, score in results}

    # 3. Aggregate
    #passage_score = aggregate_fact_scores(fact_scores, method=agg_method)
    #print(f"Generated passage scores, the time gone is {time.time() - start_time}")

    return fact_scores


with open("../test/questions.txt", "r") as f:
    for _ in range(1):
        prompt = f.readline()
        response = call_llm(prompt, temperature=0.0)

        results = None

        async def main():
            results = await fact_selfcheck_pipeline(prompt=prompt,response_text=response, method="kg", agg_method="max", n_samples=5)

            print(results)
        asyncio.run(main())


