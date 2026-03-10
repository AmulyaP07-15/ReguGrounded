# TEST retriever.py

def retrieve(query: str, top_k: int = 3):
    """
    mock retrieval function for testing reasoning pipeline
    """

    mock_chunks = [
        {
            "text": "NYC Local Law 144 requires employers using automated employment decision tools to conduct annual bias audits performed by an independent auditor.",
            "metadata": {
                "law_name": "NYC Local Law 144",
                "article_number": None,
                "section_number": "20-871",
                "chunk_id": "nyc_144_1"
            },
            "score": 0.92
        },
        {
            "text": "The EU AI Act classifies AI systems used in employment decisions as high-risk systems requiring risk management and data governance procedures.",
            "metadata": {
                "law_name": "EU AI Act",
                "article_number": "10",
                "section_number": None,
                "chunk_id": "eu_ai_10"
            },
            "score": 0.89
        }
    ]

    return mock_chunks[:top_k]