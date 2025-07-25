import difflib
import spacy
from neo4j import GraphDatabase
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import os

nlp = spacy.load("en_core_web_sm")

# Hardcoded knowledge base
knowledge_base = {
    "Flu": {
        "symptoms": ["Fever", "Cough", "Headache"],
        "prob": {"Fever": 0.8, "Cough": 0.7, "Headache": 0.6}
    },
    "COVID-19": {
        "symptoms": ["Fever", "Cough", "Shortness of Breath", "Loss of Smell"],
        "prob": {"Fever": 0.85, "Cough": 0.65, "Shortness of Breath": 0.75, "Loss of Smell": 0.8}
    },
    "Cold": {
        "symptoms": ["Sneezing", "Runny Nose", "Cough"],
        "prob": {"Sneezing": 0.9, "Runny Nose": 0.8, "Cough": 0.4}
    },
    "Malaria": {
        "symptoms": ["Fever", "Chills", "Sweating"],
        "prob": {"Fever": 0.9, "Chills": 0.85, "Sweating": 0.75}
    },
    "Typhoid": {
        "symptoms": ["Fever", "Headache", "Abdominal Pain"],
        "prob": {"Fever": 0.8, "Headache": 0.7, "Abdominal Pain": 0.6}
    },
    "Asthma": {
        "symptoms": ["Cough", "Shortness of Breath", "Wheezing"],
        "prob": {"Cough": 0.7, "Shortness of Breath": 0.8, "Wheezing": 0.9}
    },
    "Allergy": {
        "symptoms": ["Sneezing", "Runny Nose", "Itchy Eyes"],
        "prob": {"Sneezing": 0.8, "Runny Nose": 0.7, "Itchy Eyes": 0.6}
    },
    "Dengue": {
        "symptoms": ["Fever", "Headache", "Muscle Pain", "Rash"],
        "prob": {"Fever": 0.95, "Headache": 0.85, "Muscle Pain": 0.8, "Rash": 0.7}
    },
    "Pneumonia": {
        "symptoms": ["Fever", "Cough", "Chest Pain", "Shortness of Breath"],
        "prob": {"Fever": 0.9, "Cough": 0.8, "Chest Pain": 0.7, "Shortness of Breath": 0.6}
    },
    "Tuberculosis": {
        "symptoms": ["Cough", "Weight Loss", "Night Sweats", "Fever"],
        "prob": {"Cough": 0.85, "Weight Loss": 0.75, "Night Sweats": 0.65, "Fever": 0.9}
    }
}

# Neo4j setup
uri = "bolt://localhost:7687"
username = "neo4j"
password = "arfaaaisha"
driver = GraphDatabase.driver(uri, auth=(username, password))
    
def load_knowledge_from_textfile(filepath=None):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), 'knowledge.txt')
        #print(f"Loading knowledge from: {filepath}")  # for directory debugging
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if not line or 'has symptoms' not in line.lower():
                    continue

                doc = nlp(line)
                disease = None
                symptoms = []

                # Simple logic: first proper noun before "has symptoms" is disease
                for token in doc:
                    if token.text.lower() == "has":
                        disease = line[:token.idx].strip().title()
                        break

                # Extract symptoms using pattern after "has symptoms"
                if "has symptoms" in line.lower():
                    after_phrase = line.lower().split("has symptoms", 1)[1]
                    symptoms = [s.strip().title() for s in after_phrase.replace('.', '').split(',') if s.strip()]

                if disease and symptoms:
                    if disease not in knowledge_base:
                        knowledge_base[disease] = {'symptoms': symptoms, 'prob': {}}
                    else:
                        for s in symptoms:
                            if s not in knowledge_base[disease]['symptoms']:
                                knowledge_base[disease]['symptoms'].append(s)

                    for symptom in symptoms:
                        if symptom not in knowledge_base[disease]['prob']:
                            knowledge_base[disease]['prob'][symptom] = 0.7  # Default probability
    except Exception as e:
        print(f"Error loading knowledge.txt: {e}")


def handle_complex_sentence_input(sentence):
    if 'therefore' not in sentence.lower() or 'has' not in sentence.lower():
        print("Invalid format. Expected: '<Name> has Symptom1 and Symptom2, therefore they might have Disease1 or Disease2'")
        return [], []

    try:
        before, after = sentence.lower().split('therefore')
        name_part, symptoms_part = before.split('has')
        raw_symptoms = symptoms_part.replace('.', '').strip().split('and')
        symptoms = [s.strip().title() for s in raw_symptoms]

        raw_diseases = after.replace('they might have', '').replace('he might have', '').replace('she might have', '').replace('.', '').strip().split('or')
        diseases = [d.strip().title() for d in raw_diseases]

        return symptoms, diseases
    except Exception as e:
        print(f"Error parsing sentence: {e}")
        return [], []

def get_all_symptoms():
    all_symptoms = set()
    for disease_data in knowledge_base.values():
        all_symptoms.update(disease_data["symptoms"])
    return list(all_symptoms)

def validate_symptom(symptom, all_symptoms):
    if symptom in all_symptoms:
        return symptom
    matches = difflib.get_close_matches(symptom, all_symptoms, n=1, cutoff=0.6)
    if matches:
        choice = input(f"Did you mean '{matches[0]}'? (yes/no): ").strip().lower()
        if choice == 'yes':
            return matches[0]
    print("Symptom not recognized. Please enter a valid symptom.")
    return None

def build_bn_model(valid_symptoms):
    edges = []
    cpds = []
    diseases = list(knowledge_base.keys())

    for disease in diseases:
        for symptom in knowledge_base[disease]['symptoms']:
            if symptom in valid_symptoms:
                edges.append((symptom, disease))

    model = DiscreteBayesianNetwork(edges)

    used_symptoms = set()
    for edge in edges:
        symptom = edge[0]
        if symptom not in used_symptoms:
            cpds.append(TabularCPD(variable=symptom, variable_card=2, values=[[0.5], [0.5]]))
            used_symptoms.add(symptom)

    for disease in diseases:
        relevant_symptoms = [s for s in knowledge_base[disease]['symptoms'] if s in valid_symptoms]
        if not relevant_symptoms:
            continue
        prob_true = []
        prob_false = []

        for i in range(2**len(relevant_symptoms)):
            bin_input = format(i, f"0{len(relevant_symptoms)}b")
            p = 1.0
            for idx, bit in enumerate(bin_input):
                s = relevant_symptoms[idx]
                prob = knowledge_base[disease]['prob'][s]
                p *= prob if bit == '1' else (1 - prob)
            prob_true.append(p)
            prob_false.append(1 - p)

        cpd = TabularCPD(
            variable=disease,
            variable_card=2,
            values=[prob_false, prob_true],
            evidence=relevant_symptoms,
            evidence_card=[2]*len(relevant_symptoms)
        )
        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()
    return model

def infer_probabilities(model, symptoms):
    inference = VariableElimination(model)
    result = {}
    diseases = [var for var in model.nodes() if var not in symptoms]
    evidence = {symptom: 1 for symptom in symptoms}

    for disease in diseases:
        try:
            q = inference.query(variables=[disease], evidence=evidence)
            result[disease] = round(q.values[1], 2)  # P(disease=1)
        except:
            result[disease] = 0.0
    return result

def send_to_neo4j(symptoms, diagnosis):
    with driver.session() as session:
        for symptom in symptoms:
            session.run("MERGE (s:Symptom {name: $symptom})", symptom=symptom)
        for disease, prob in diagnosis.items():
            if prob > 0:
                session.run(
                    "MERGE (d:Disease {name: $disease}) SET d.severity = 'High', d.location = 'Global', d.age_group = 'Adult'",
                    disease=disease)
                for symptom in symptoms:
                    session.run("""
                    MATCH (s:Symptom {name: $symptom})
                    MATCH (d:Disease {name: $disease})
                    MERGE (s)-[r:POSSIBLY_INDICATES {probability: $prob}]->(d)
                    """, symptom=symptom, disease=disease, prob=prob)

def main():
    load_knowledge_from_textfile()  # Load additional knowledge using NLP (Task 4)
    all_symptoms = get_all_symptoms()
    valid_symptoms = []

    use_complex = input("Would you like to enter a complex sentence instead? (yes/no): ").strip().lower()
    if use_complex == 'yes':
        sentence = input("Enter the complex sentence: ")
        symptoms, _ = handle_complex_sentence_input(sentence)
        for s in symptoms:
            if s in all_symptoms:
                valid_symptoms.append(s)
            else:
                vs = validate_symptom(s, all_symptoms)
                if vs:
                    valid_symptoms.append(vs)
    else:
        print("Enter your symptoms one at a time. Type 'done' when finished:")
        while True:
            s = input("Symptom: ").strip().title()
            if s.lower() == 'done':
                break
            vs = validate_symptom(s, all_symptoms)
            if vs:
                valid_symptoms.append(vs)

    if not valid_symptoms:
        print("No valid symptoms entered. Exiting.")
        return

    model = build_bn_model(valid_symptoms)
    diagnosis = infer_probabilities(model, valid_symptoms)
    print("\nDiagnosis:")
    print(diagnosis)

    send_to_neo4j(valid_symptoms, diagnosis)
    print("\nResults have been sent to Neo4j. You can view the knowledge graph now.")

if __name__ == '__main__':
    main()
