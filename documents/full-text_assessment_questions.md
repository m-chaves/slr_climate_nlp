Text after “|” indicates the column’s format. In particular, “updatable multi-select” indicates that the column is categorical, where multiple elements can be selected and where we should be able to change and add new labels as we go through the papers. 
Elements between parenthesis indicate the (initial) labels to include.
No question should have blanks in the end:  "not reported", "not applicable (NA)" or "unclear" should be options to select depending on the question. 
An “*” next to a label indicates that if that label is selected, then the child questions should be answered. If the label with the asterix is not selected then all child questions should be NA. 
All questions are answered at paper level except for Section 7 Data. In this case it is answered by dataset within each paper. 

1. Climate
1.1. What is the climate domain? (e.g., climate denial, wildfires) | Updatable Multi-select 

2. Misinformation. Is misinformation present? | Single-select (yes*/no) 
2.1. Is there a definition of misinformation? | Single select (yes*/no/NA)
2.1.1. Cite the definition of misinformation stated in the study. | Free text (NA)
2.1.2. State the source of the definition (e.g., (Zhang. et al, 2026 ), UN) | Free text (NA/not reported)
2.2. What is the misinformation domain? (e.g., conspiracy theories, climate denial) | Updatable Multi-select (NA) 
2.3. How is truth status established? (e.g., professional fact checkers, scientific-expert judgement, source-level credibility heuristic, crowd labels, LLM judgement) | Updatable Multi-select (NA)
2.4. What are the findings regarding misinformation? | Free text (NA)

3. Polarization. Is polarization present? Is polarization implicit or explicit? It is considered implicit when the study does not mention polarization but it measures things that are adjacent (network analysis, changes in discourse, echo chambers… need better phrasing here: Studied via adjacent or proxy constructs without naming it. | Single select (yes (explicit)* / yes (implicit)* / no)  
3.1. Is there a definition of polarization? | Single-select (yes*/no)
3.1.1. Cite the definition of polarization stated in the study. | Free text (NA)
3.1.2. State the source of the definition (e.g., (Zhang. et al, 2026 ), UN) | Free text (NA/not reported)
3.2. What is the polarization domain? (e.g., political polarization, affective polarization) | Updatable Multi-select (NA)
3.3. How is polarization measured? (e.g., network modularity/community detection, random-walk controversy or similar graph scores, opinion-distribution statistics, embedding/semantic distance, outgroup sentiment or toxicity, stance-label distribution, retweet/interaction-network structure). | Updatable Multi-select (NA)
3.4. What are the findings regarding polarization? | Free text (NA)
3.5. What is the change in polarization? | Updatable Single-select (increasing / decreasing / stable / mixed / not assessed over time / NA)

4. Motivations, methodology, actionability, alignment
4.1. Cite the motivation(s) of the study? Why do they do what they do? | Free text (Not reported)
4.1.1 Under which categories those motivations belong? | Updatable Multi-select (ML model advance / dataset creation / empirical understanding of discourse / theory testing / intervention or countermeasure design / policy or practitioner support / NA )
4.1.2. Is there a clear motivation in terms of climate, misinformation, or polarization besides just an improvement on machine learning metrics? | Single-select (yes*/no)
4.2. Is there a theoretical basis for methodological approaches? That is,  does the methodology they define align with the construct they measure? E.g., the definition of polarization matches the metrics of polarization they propose. | Single-select Free text (0 = construct never defined, measure unjustified / 1 = construct defined but measure not linked to the definition / 2 = partial or implicit link / 3 = explicit link, measure justified against the stated definition / 4 = explicit link plus validation (human validation, sensitivity analysis, comparison of alternative operationalizations) / )
4.2.1.Cite and/or explain why. | Free text
4.3. Is the methodology actionable? (“Actionability refers to the degree to which the method proposed enables decision-making or intervention” definition modified from (Delobelle et al., 2024)). | Single-select (0 = none / 1 = generic "this could be useful for practitioners" gesture /  2 = concrete implications for a named actor and decision /  3 = an artifact usable for decision-making (deployed tool, dashboard, released system, intervention design); 4 = evaluated in a real decision context or with the intended users. )
4.3.1. Cite and/or explain why. | Free text

5. Stakeholders.
5.1. Are stakeholders (e.g., fact checkers, policy makers) mentioned? | Single select (0 not mentioned / 1 generic "this could be useful for practitioners" gesture* / 2 named as intended users with specifics (you can point to a place in the study where the stakeholder requirements constrains the study)* / 3 consulted (interviews, requirements gathering, feedback)* / 4 involved in design or evaluation (participatory, co-design, they evaluated outputs)* / 5 co-authors*)
5.1.1. Who are the stakeholders? | Updatable Multi-select (fact-checkers / policy makers / NA) 
5.1.2. Cite the stakeholders. | Free text	

6. Conclusions and findings
6.1. Is there a clear conclusion in terms of climate, misinformation, or polarization besides just an improvement on machine learning metrics? | Single-select (yes*/no)
6.2. Do the motivations and conclusions match? | Single select (0 = conclusions unrelated to motivation; 1 = partially addressed; 2 = addressed; 3 = addressed with explicit reflection on what was and wasn't achieved) 

7. Data
7.1.What kind of data do they use? | Free text
7.2 What is the modality? |  Updatable Multi-select (text / images / video)
7.3 What is the name of the dataset? | Free text 
7.4 From which platform was the data retrieved?  Updatable Multi-select (Twitter / Reddit / YouTube)
7.5 In which language? | Updatable Multi-select (English)
7.6 What is the thematic? | Updatable Multi-select 
7.7 Is it a new dataset? | Single select (yes / no / it is an extension)
7.8. What is the time span starting? | Integer
7.9. What is the time span ending? | Integer
7.10. Did they perform annotations or labelled data?  | Single-select (yes* / no)
7.10.1. Who performed the annotations ? | Updatable multi-select (papers authors / domain experts / trained annotators / Crowdworkers / Volunteers / Study participants (Survey respondents labeling their own beliefs or content) / LLM or generative model* / Non-generative classifier* / Derived from data (Labels come from a signal already in the data) / Not reported / NA)
7.10.1.1. How were LLMs or other models used for annotations? | Updatable Multi-select (prelabel (the model pre-labels with human review) / voter (Treated as one annotator among humans, aggregated by vote or averaging) / sole annotator / Resolves disagreements between human annotators / unclear / NA )
7.11. What is the ML task the dataset is intended for? | Updatable multi-select (stance detection or classification / sentiment / claim detection or classification / topic modeling / summarization / generation / retrieval / network analysis / time-series analysis / regression or statistical inference / not reported / unclear )

8. ML and computational methods
8.1. Which ML task(s) do they tackle? | Updatable multi-select (stance detection or classification / sentiment / claim detection or classification / topic modeling / summarization / generation / retrieval / network analysis / time-series analysis / regression or statistical inference / not reported / unclear )
8.2. Which computational method do they use? | Updatable multi-select (Lexicon/rule-based (VADER, LIWC, custom dictionaries) / Classical ML (LR, SVM, NB, RF, gradient boosting) / Statistical/econometric (GLM, mixed models, time series) / Topic models (LDA, NMF, STM, BERTopic) / Static embeddings (word2vec, GloVe, fastText) / Non-transformer neural (CNN, RNN/LSTM) / Encoder-only PLMs (BERT, RoBERTa, XLM-R, ClimateBERT) / Encoder–decoder (T5, BART) / Decoder-only LLMs / Graph/network methods (community detection, centrality, GNNs) / Multimodal models (CLIP, VLMs) / Unclear )
8.3. Which ML subtechniques or frameworks are used? | Updatable Multi-select (zero-shot / few-shot / Chain of Thought / prompt-engineered / instruction tuning / (LoRA) fine-tuning / regular fine-tuning / RAG / knowledge distillation / role-played personas / agentic / NA) 
