#!/usr/bin/env python3
"""
Glossary Population Script

This script adds 60 new terms and definitions to the ArangoDB glossary collection.
Each term is embedded using the Nomic ModernBert model to enable semantic search.
"""

import asyncio
import random
from arango import ArangoClient
from loguru import logger

from agent_tools.cursor_rules.core.glossary import format_for_embedding
from agent_tools.cursor_rules.core.cursor_rules import generate_embedding

# Configuration
ARANGO_HOST = "http://localhost:8529"
DB_NAME = "test_semantic_term_definition_db"
COLLECTION_NAME = "test_semantic_glossary"
USERNAME = "root"
PASSWORD = "openSesame"

# 60 glossary terms and definitions covering various AI, ML, and tech topics
GLOSSARY_ENTRIES = [
    # Machine Learning Fundamentals
    {
        "term": "Decision Tree",
        "definition": "A tree-like model of decisions where each internal node represents a feature, each branch represents a decision rule, and each leaf node represents an outcome.",
        "category": "Machine Learning",
        "related_terms": ["Random Forest", "Gradient Boosting", "Machine Learning"]
    },
    {
        "term": "Random Forest",
        "definition": "An ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes or mean prediction of the individual trees.",
        "category": "Machine Learning",
        "related_terms": ["Decision Tree", "Ensemble Learning", "Bagging"]
    },
    {
        "term": "Support Vector Machine",
        "definition": "A supervised learning model that analyzes data for classification and regression analysis, using hyperplanes to separate data points in high-dimensional space.",
        "category": "Machine Learning",
        "related_terms": ["Kernel Trick", "Linear Classification", "Supervised Learning"]
    },
    {
        "term": "K-means Clustering",
        "definition": "An unsupervised learning algorithm that partitions n observations into k clusters where each observation belongs to the cluster with the nearest mean.",
        "category": "Machine Learning",
        "related_terms": ["Unsupervised Learning", "Clustering", "Vector Quantization"]
    },
    {
        "term": "Logistic Regression",
        "definition": "A statistical model that models the probability of an event taking place by having the log-odds for the event be a linear combination of one or more independent variables.",
        "category": "Machine Learning",
        "related_terms": ["Linear Regression", "Classification", "Supervised Learning"]
    },
    
    # Deep Learning
    {
        "term": "Convolutional Neural Network",
        "definition": "A class of deep neural networks most commonly applied to analyzing visual imagery, using convolutional layers to filter inputs for useful information.",
        "category": "Deep Learning",
        "related_terms": ["Deep Learning", "Neural Network", "Computer Vision"]
    },
    {
        "term": "Recurrent Neural Network",
        "definition": "A class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence, allowing the network to exhibit temporal dynamic behavior.",
        "category": "Deep Learning",
        "related_terms": ["LSTM", "Neural Network", "Sequence Modeling"]
    },
    {
        "term": "LSTM",
        "definition": "Long Short-Term Memory networks are a type of RNN capable of learning long-term dependencies, using a cell state and various gates to regulate information flow.",
        "category": "Deep Learning",
        "related_terms": ["Recurrent Neural Network", "GRU", "Sequence Modeling"]
    },
    {
        "term": "GRU",
        "definition": "Gated Recurrent Unit is a gating mechanism in recurrent neural networks, similar to LSTM but with fewer parameters, using update and reset gates.",
        "category": "Deep Learning",
        "related_terms": ["LSTM", "Recurrent Neural Network", "Deep Learning"]
    },
    {
        "term": "Transformer",
        "definition": "A deep learning model architecture that relies entirely on self-attention mechanisms without using recurrence or convolution, enabling more parallelization and better handling of long-range dependencies.",
        "category": "Deep Learning",
        "related_terms": ["Self-Attention", "Neural Network", "Natural Language Processing"]
    },
    
    # Natural Language Processing
    {
        "term": "Self-Attention",
        "definition": "A mechanism in neural networks that weighs the importance of different elements in a sequence for a specific task, allowing the model to focus on relevant parts of the input.",
        "category": "Natural Language Processing",
        "related_terms": ["Transformer", "Neural Network", "Attention Mechanism"]
    },
    {
        "term": "BERT",
        "definition": "Bidirectional Encoder Representations from Transformers, a transformer-based model pre-trained on large text corpora that can be fine-tuned for specific NLP tasks.",
        "category": "Natural Language Processing",
        "related_terms": ["Transformer", "Pre-training", "Fine-tuning"]
    },
    {
        "term": "GPT",
        "definition": "Generative Pre-trained Transformer, a language model based on the transformer architecture, trained on large amounts of text data to generate human-like text.",
        "category": "Natural Language Processing",
        "related_terms": ["Transformer", "Language Model", "Text Generation"]
    },
    {
        "term": "Word Embedding",
        "definition": "A technique where words are represented as vectors in a continuous vector space, capturing semantic and syntactic similarities between words.",
        "category": "Natural Language Processing",
        "related_terms": ["Embedding", "Word2Vec", "GloVe"]
    },
    {
        "term": "Tokenization",
        "definition": "The process of breaking down text into smaller units such as words, subwords, or characters, which can then be processed by NLP algorithms.",
        "category": "Natural Language Processing",
        "related_terms": ["NLP", "Text Processing", "Preprocessing"]
    },
    
    # Computer Vision
    {
        "term": "Object Detection",
        "definition": "A computer vision technique that involves detecting instances of semantic objects of a certain class in digital images and videos.",
        "category": "Computer Vision",
        "related_terms": ["Computer Vision", "CNN", "YOLO"]
    },
    {
        "term": "Semantic Segmentation",
        "definition": "The process of assigning a class label to each pixel in an image, enabling understanding of what's in the image at the pixel level.",
        "category": "Computer Vision",
        "related_terms": ["Computer Vision", "Image Processing", "U-Net"]
    },
    {
        "term": "YOLO",
        "definition": "You Only Look Once, a real-time object detection system that processes the entire image in a single pass through the neural network, making it very fast.",
        "category": "Computer Vision",
        "related_terms": ["Object Detection", "CNN", "Computer Vision"]
    },
    {
        "term": "Feature Extraction",
        "definition": "The process of reducing the amount of resources required to describe a large set of data by identifying relevant features that can be used for analysis.",
        "category": "Computer Vision",
        "related_terms": ["CNN", "Machine Learning", "Dimensionality Reduction"]
    },
    {
        "term": "Transfer Learning",
        "definition": "A machine learning method where a model developed for a task is reused as the starting point for a model on a second task, often used in computer vision and NLP.",
        "category": "Machine Learning",
        "related_terms": ["Fine-tuning", "Pre-training", "Domain Adaptation"]
    },
    
    # Reinforcement Learning
    {
        "term": "Reinforcement Learning",
        "definition": "A type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize some notion of cumulative reward.",
        "category": "Reinforcement Learning",
        "related_terms": ["Q-Learning", "Deep Q-Network", "Agent"]
    },
    {
        "term": "Q-Learning",
        "definition": "A model-free reinforcement learning algorithm to learn the value of an action in a particular state, used to find optimal action-selection policies.",
        "category": "Reinforcement Learning",
        "related_terms": ["Reinforcement Learning", "Deep Q-Network", "Policy"]
    },
    {
        "term": "Policy Gradient",
        "definition": "A reinforcement learning approach that directly optimizes the policy by estimating the gradient of expected rewards with respect to the policy parameters.",
        "category": "Reinforcement Learning",
        "related_terms": ["Reinforcement Learning", "Actor-Critic", "REINFORCE"]
    },
    {
        "term": "Deep Q-Network",
        "definition": "A combination of Q-Learning and deep neural networks, allowing agents to learn from high-dimensional sensory inputs and achieve superhuman performance on many tasks.",
        "category": "Reinforcement Learning",
        "related_terms": ["Q-Learning", "Reinforcement Learning", "Deep Learning"]
    },
    {
        "term": "Markov Decision Process",
        "definition": "A mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker.",
        "category": "Reinforcement Learning",
        "related_terms": ["Reinforcement Learning", "State", "Action"]
    },
    
    # Database and Data Storage
    {
        "term": "Graph Database",
        "definition": "A database that uses graph structures with nodes, edges, and properties to represent and store data, optimized for complex relationships between data points.",
        "category": "Databases",
        "related_terms": ["Database", "Neo4j", "ArangoDB"]
    },
    {
        "term": "Document Store",
        "definition": "A NoSQL database designed for storing, retrieving, and managing document-oriented information, typically encoded in formats like JSON, BSON, or XML.",
        "category": "Databases",
        "related_terms": ["NoSQL", "MongoDB", "CouchDB"]
    },
    {
        "term": "Key-Value Store",
        "definition": "A data storage paradigm designed for storing, retrieving, and managing associative arrays, a collection of key-value pairs where the key serves as a unique identifier.",
        "category": "Databases",
        "related_terms": ["Redis", "NoSQL", "Database"]
    },
    {
        "term": "Columnar Database",
        "definition": "A database management system that stores data tables by column rather than by row, providing efficient storage and query performance for analytical workloads.",
        "category": "Databases",
        "related_terms": ["Database", "BigQuery", "Cassandra"]
    },
    {
        "term": "Data Warehouse",
        "definition": "A system used for reporting and data analysis, considered a core component of business intelligence, storing current and historical data from multiple sources.",
        "category": "Databases",
        "related_terms": ["Database", "Business Intelligence", "ETL"]
    },
    
    # ML Ops and Infrastructure
    {
        "term": "Containerization",
        "definition": "A lightweight form of virtualization that involves packaging an application and its dependencies, configurations, and other necessary parts into a container.",
        "category": "Infrastructure",
        "related_terms": ["Docker", "Kubernetes", "DevOps"]
    },
    {
        "term": "CI/CD",
        "definition": "Continuous Integration and Continuous Deployment, a method to frequently deliver apps to customers by introducing automation into the stages of app development.",
        "category": "Infrastructure",
        "related_terms": ["DevOps", "Automation", "Pipeline"]
    },
    {
        "term": "Feature Store",
        "definition": "A centralized repository that enables data scientists to find and reuse features, ensuring consistency across training and inference.",
        "category": "MLOps",
        "related_terms": ["MLOps", "Machine Learning", "Data Science"]
    },
    {
        "term": "Model Registry",
        "definition": "A centralized repository for tracking and managing machine learning models throughout their lifecycle, from development to deployment.",
        "category": "MLOps",
        "related_terms": ["MLOps", "Machine Learning", "Model Governance"]
    },
    {
        "term": "Kubernetes",
        "definition": "An open-source platform for automating deployment, scaling, and operations of application containers across clusters of hosts.",
        "category": "Infrastructure",
        "related_terms": ["Container", "Docker", "Orchestration"]
    },
    
    # Data Science and Statistics
    {
        "term": "Dimensionality Reduction",
        "definition": "The process of reducing the number of random variables under consideration by obtaining a set of principal variables, often used for data visualization and machine learning.",
        "category": "Data Science",
        "related_terms": ["PCA", "t-SNE", "Feature Selection"]
    },
    {
        "term": "Principal Component Analysis",
        "definition": "A technique used to emphasize variation and bring out strong patterns in a dataset, often used to make data easy to explore and visualize.",
        "category": "Data Science",
        "related_terms": ["Dimensionality Reduction", "SVD", "Statistics"]
    },
    {
        "term": "Bayesian Inference",
        "definition": "A method of statistical inference in which Bayes' theorem is used to update the probability for a hypothesis as more evidence or information becomes available.",
        "category": "Statistics",
        "related_terms": ["Statistics", "Probability", "Bayes' Theorem"]
    },
    {
        "term": "Hypothesis Testing",
        "definition": "A statistical method for making decisions using data, testing a claim about a parameter in a population using a sample data set.",
        "category": "Statistics",
        "related_terms": ["Statistics", "P-value", "Null Hypothesis"]
    },
    {
        "term": "Feature Engineering",
        "definition": "The process of using domain knowledge to select and transform the most relevant variables from raw data for creating machine learning models.",
        "category": "Data Science",
        "related_terms": ["Machine Learning", "Feature Selection", "Data Preprocessing"]
    },
    
    # Specialized ML Techniques
    {
        "term": "Anomaly Detection",
        "definition": "The identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data.",
        "category": "Machine Learning",
        "related_terms": ["Outlier Detection", "Unsupervised Learning", "Fraud Detection"]
    },
    {
        "term": "Ensemble Learning",
        "definition": "A machine learning paradigm where multiple models are trained to solve the same problem and combined to get better results than any single model.",
        "category": "Machine Learning",
        "related_terms": ["Random Forest", "Boosting", "Bagging"]
    },
    {
        "term": "Gradient Boosting",
        "definition": "A machine learning technique that produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.",
        "category": "Machine Learning",
        "related_terms": ["Ensemble Learning", "XGBoost", "AdaBoost"]
    },
    {
        "term": "Active Learning",
        "definition": "A special case of machine learning in which a learning algorithm can interactively query a user or other information source to label new data points.",
        "category": "Machine Learning",
        "related_terms": ["Machine Learning", "Semi-Supervised Learning", "Human-in-the-loop"]
    },
    {
        "term": "Semi-Supervised Learning",
        "definition": "A learning paradigm concerned with the study of how to leverage unlabeled data, typically a small amount of labeled data with a large amount of unlabeled data.",
        "category": "Machine Learning",
        "related_terms": ["Supervised Learning", "Unsupervised Learning", "Machine Learning"]
    },
    
    # AI Ethics and Fairness
    {
        "term": "Algorithmic Bias",
        "definition": "Systematic and repeatable errors in a computer system that create unfair outcomes, such as privileging one arbitrary group of users over others.",
        "category": "AI Ethics",
        "related_terms": ["Fairness", "Ethics", "Machine Learning"]
    },
    {
        "term": "Fairness Metrics",
        "definition": "Quantitative tools to assess whether an algorithmic system treats different groups of people fairly, often measuring disparities in model outcomes across demographic groups.",
        "category": "AI Ethics",
        "related_terms": ["Algorithmic Bias", "Ethics", "Evaluation"]
    },
    {
        "term": "Explainable AI",
        "definition": "Artificial intelligence systems whose actions can be easily understood by humans, providing transparency and interpretability for AI decisions.",
        "category": "AI Ethics",
        "related_terms": ["Interpretability", "Transparency", "Machine Learning"]
    },
    {
        "term": "Differential Privacy",
        "definition": "A system for publicly sharing information about a dataset by describing patterns of groups within the dataset while withholding information about individuals.",
        "category": "AI Ethics",
        "related_terms": ["Privacy", "Data Protection", "Anonymization"]
    },
    {
        "term": "Model Interpretability",
        "definition": "The ability to explain or present a model's decision-making process in terms that humans can understand, crucial for trust and adoption of AI systems.",
        "category": "AI Ethics",
        "related_terms": ["Explainable AI", "Feature Importance", "Black Box Models"]
    },
    
    # Miscellaneous
    {
        "term": "AutoML",
        "definition": "Automated Machine Learning, the process of automating the end-to-end process of applying machine learning to real-world problems.",
        "category": "Machine Learning",
        "related_terms": ["Neural Architecture Search", "Hyperparameter Optimization", "Machine Learning"]
    },
    {
        "term": "Multi-modal Learning",
        "definition": "A machine learning approach that processes and relates information from multiple modalities, such as text, image, audio, and video.",
        "category": "Machine Learning",
        "related_terms": ["Deep Learning", "Computer Vision", "Natural Language Processing"]
    },
    {
        "term": "Few-shot Learning",
        "definition": "A type of machine learning where a model must generalize to new classes or tasks with only a few examples of each class.",
        "category": "Machine Learning",
        "related_terms": ["Meta Learning", "Transfer Learning", "Zero-shot Learning"]
    },
    {
        "term": "Meta Learning",
        "definition": "A learning paradigm where an algorithm learns how to learn, capable of adapting to new tasks quickly with little data.",
        "category": "Machine Learning",
        "related_terms": ["Few-shot Learning", "Transfer Learning", "Neural Architecture Search"]
    },
    {
        "term": "Federated Learning",
        "definition": "A machine learning technique that trains an algorithm across multiple decentralized devices holding local data samples, without exchanging them.",
        "category": "Machine Learning",
        "related_terms": ["Privacy", "Distributed Computing", "Edge Computing"]
    }
]

async def main():
    """Main function to populate the glossary collection with embedded terms."""
    # Connect to ArangoDB
    client = ArangoClient(hosts=ARANGO_HOST)
    db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)
    
    # Get or create the collection
    if not db.has_collection(COLLECTION_NAME):
        collection = db.create_collection(COLLECTION_NAME)
        logger.info(f"Created collection: {COLLECTION_NAME}")
    else:
        collection = db.collection(COLLECTION_NAME)
        logger.info(f"Using existing collection: {COLLECTION_NAME}")
    
    # Get existing terms to avoid duplicates
    existing_terms = set()
    cursor = collection.all()
    for doc in cursor:
        if "term" in doc:
            existing_terms.add(doc["term"].lower())
    
    logger.info(f"Found {len(existing_terms)} existing terms in the collection")
    
    # Process and insert new terms
    added_count = 0
    skipped_count = 0
    
    for entry in GLOSSARY_ENTRIES:
        term = entry["term"]
        
        # Skip if term already exists
        if term.lower() in existing_terms:
            logger.debug(f"Skipping existing term: {term}")
            skipped_count += 1
            continue
        
        # Generate embedding for the term and definition
        text_to_embed = format_for_embedding(term, entry["definition"])
        embedding_result = generate_embedding(text_to_embed)
        
        if not embedding_result or "embedding" not in embedding_result:
            logger.warning(f"Failed to generate embedding for term: {term}")
            continue
        
        # Add embedding to the entry
        entry["embedding"] = embedding_result["embedding"]
        entry["source"] = "script_population"
        
        # Insert into collection
        try:
            collection.insert(entry)
            logger.info(f"Added term with embedding: {term}")
            added_count += 1
            # Add to existing terms set to avoid duplicates if the same term appears multiple times
            existing_terms.add(term.lower())
        except Exception as e:
            logger.error(f"Error inserting term {term}: {e}")
    
    logger.info(f"Population complete: Added {added_count} terms, skipped {skipped_count} terms")
    
    # Get total count after population
    total_count = collection.count()
    logger.info(f"Total terms in collection: {total_count}")

if __name__ == "__main__":
    asyncio.run(main()) 