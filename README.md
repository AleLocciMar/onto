# OntoML: Integração Semântica para Diagnóstico de Falhas em ML

Este repositório contém os experimentos e a modelagem ontológica da minha tese de doutorado, focada no modelo **RIPR** (Result, Infection, Propagation, Residual). O objetivo é utilizar ontologias para detectar e diagnosticar falhas em pipelines de Machine Learning de forma autônoma.

## 🚀 O Experimento Atual
O foco desta etapa foi a detecção de instabilidade de gradiente causada pelo uso de **Stochastic Gradient Descent (Batch Size = 1)**.

- **Problema:** Acurácia abaixo de 50% e ruído excessivo nas métricas de treino.
- **Solução Semântica:** Mapeamento dos resultados do TensorFlow para a ontologia **MLTO** utilizando a biblioteca `owlready2`.

## 🧠 Arquitetura do Conhecimento
O projeto utiliza a ontologia **MLTO** (Machine Learning Taxonomy Ontology) para classificar os experimentos:
- **Infection:** Identificada quando o Batch Size é inadequado.
- **Symptom:** `GradientInstability` detectada automaticamente via inferência.
- **State:** Transição do modelo para o estado `Infected`.

## 📂 Estrutura do Repositório
- `Onto.ipynb`: Notebook Jupyter com o treinamento do modelo (TensorFlow) e a integração ontológica.
- `tese_mlto_alexandre.owl`: Arquivo da ontologia populada com as instâncias dos experimentos (formato RDF/XML).
- `.gitignore`: Configurações para manter o repositório limpo.

## 🛠️ Como Executar
1. Instale as dependências:
   ```bash
   pip install tensorflow owlready2 pandas matplotlib
