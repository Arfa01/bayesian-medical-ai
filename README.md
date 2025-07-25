# Bayesian Network–Based Medical Diagnosis System

A smart AI-powered system that predicts probable diseases based on user-provided symptoms using **Bayesian inference** and visualizes relationships through a **Neo4j knowledge graph**.

---

## Overview

This system integrates **Natural Language Processing (spaCy)**, **Graph Databases (Neo4j)**, and **Bayesian Networks (pgmpy)** to simulate a basic diagnostic tool. Users can input symptoms in free text or structured form, and the system returns the top possible diseases with calculated probabilities.

---

## Features

- **Free-Form Symptom Input** using NLP
  - Accepts inputs like `"Ali has fever and cough..."`
  - Uses `spaCy` to extract structured symptoms

- **Bayesian Inference with pgmpy**
  - Builds a dynamic Bayesian network
  - Calculates disease probabilities based on given symptoms

- **Fuzzy Matching for Symptoms**
  - Mistyped or unknown symptoms are corrected using `difflib.get_close_matches()`

- **Neo4j Graph Representation**
  - Visualizes disease-symptom relationships
  - Automatically builds nodes and "POSSIBLY_INDICATES" relationships

- **Visual Output**
  - Plots bar/pie charts of most likely diagnoses using `matplotlib`
  - Displays top 3 most probable diseases

---

## Sample Input & Output

### Input:
Ali is coughing and has a fever. Might be flu or COVID.

### Output:

Symptoms Detected: ['fever', 'cough']
Top 3 Probable Diseases:
Flu – 0.72
COVID-19 – 0.64
Cold – 0.52

---

## 🗃️ Project Structure

bayesian-medical-ai/
├── bayesian.py # Main inference engine
├── testing-guide.txt # Example test cases & helper UI
├── knowledge.txt # Knowledge loader from text file
├── assets/ # Screenshots, diagrams
└── README.md


---

## Technologies & Tools

- **Python 3.10+**
- **pgmpy** – Bayesian Network modeling
- **Neo4j + Cypher** – Graph database & relationships
- **spaCy** – NLP symptom parsing
- **Matplotlib** – Graphical result display
- **difflib** – Fuzzy symptom correction

---

## Key Learnings

- Uncertainty modeling using probabilistic graphs
- Knowledge representation with semantic networks (Neo4j)
- Natural language processing for information extraction
- Input handling, modular code design, multi-layered integration

---

## Architecture Diagram

<img width="1200" height="600" alt="model image" src="https://github.com/user-attachments/assets/c316302d-babe-4582-a2a2-43e0a88a6b02" />

---

## 🎓 Academic Context

Built as a final project for the **Artificial Intelligence** course in 4th semester at COMSATS University. Focus areas included:
- Bayesian inference
- Graph-based modeling
- NLP and knowledge-based AI

---


## Future Work

- Add severity levels and age-based probability weights
- Integrate a real medical dataset (e.g., UCI or Kaggle)
- Expand to API-based frontend or web dashboard

---


