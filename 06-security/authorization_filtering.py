"""
Goal:
Show how authorization must be enforced before retrieval results
are returned to the user.
"""

documents = [
    {
        "text": "General AWS Glue throttling guidance.",
        "department": "engineering",
    },
    {
        "text": "Confidential payroll processing runbook.",
        "department": "finance",
    },
]

user_department = "engineering"

# Filtering happens BEFORE retrieval results reach the LLM or the user.
# Relying on the model to withhold the payroll doc after it's already in
# context is not a security boundary -- prompt injection or a jailbreak
# could still surface it, so unauthorized documents must never be fetched.
authorized_documents = [
    doc
    for doc in documents
    if doc["department"] == user_department
]

print(f"User department: {user_department}\n")

print("Authorized documents:")
for doc in authorized_documents:
    print(f"- {doc['text']}")

print(
    "\nLesson: Authorization filtering must happen before protected "
    "content is exposed to retrieval or generation."
)