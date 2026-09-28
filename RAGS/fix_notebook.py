import json
import os

file_path = 'c:/Agentic_AI_Learning/RAGS/CorrectiveRAG.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell.get('source', []):
            # Fix langchain_classic -> langchain
            line = line.replace('from langchain_classic import hub', 'from langchain import hub')
            
            # Fix docs format error
            line = line.replace('"context": docs', '"context": format_docs(docs)')
            line = line.replace("'context': docs", "'context': format_docs(docs)")
            
            new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
