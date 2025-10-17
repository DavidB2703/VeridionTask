
### Start the API Server

```bash
cd api
python app.py
```
## Modelul

Modelul final este format din urmatoarele componente:

1. **Preprocesarea Datelor**:
   - Se formeaza un string mare care contine toate datele despre o companie si se adauga la inceput textul `This is a detailed company profile`.
   - Apoi, se curata textul, cu ajutorul functiei `soft_clean`. Funcția curăță textul de linkuri, emailuri, spații și newline-uri inutile, păstrând structura de bază. Înlocuiește URL-urile și emailurile cu etichete generice și normalizează spațierea pentru un text mai curat și uniform.
2. **Sentence Transformer**:
   -  Cu ajutorul modelului `intfloat/multilingual-e5-base` de pe HuggingFace, se obțin embedding-uri pentru atat textele curățate, cat si pentru cele 220 de etichete.
   -  Mai apoi, pentru fiecare `company_embedding`, se calculează similaritatea cosinus cu fiecare dintre cele 220 de `label_embeddings` si se selectează cele mai similare 10 etichete pentru fiecare companie.
3. **Zero-shot Classification**:
   - Se folosește modelul `facebook/bart-large-mnli` pentru a obține scoruri de încredere pentru fiecare dintre cele 10 etichete selectate anterior.
   - Sunt selectate primele 3 etichete cu cele mai mari scoruri de încredere pentru fiecare companie.
## Evaluarea

- Deoarece nu exista un GroundThruth am adnotat eu 10 companii cu cate 3 laberi-uri fiecare, formand astfel un top.
Companii etichetate au fost alese random din setul de date.
- Pentru a evalua performanta modelul am creat mai mult masuri de performanta:
  - **perfection_score_tops**: Compara Top3 GT cu Top3 predictii, rasplata maxima este atunci cand pozitiile se potrivesc in top. Scorul obtinut este de *40/60* `(66.72%)`. Scorul maxim este atins doar atunci cand predictiile sunt si in aceiasi ordine ca etichetele
  - **perfection_score**: Acelasi lucru ce `perfection_score_tops`, doar ca ordinea nu mai conteaza.
  - **candidate_score**: Numara daca cel putin una din etichetele din Top3 GT se regaseste in Top3 predictii. Scorul obtinut este de *10/10* `(100%)`
  - **position_score**: Precentul de pozitii exacte. Scor obtinut este *13/30* `(43.33%)`
  - Media dintre **candidate_score** si **position_score** este de 71.66%

O alta metrica interesanta ar fi sa verific similaritatea cosinus intre embedding-urile etichetelor GT si embedding-urile etichetelor predictate.

## Puncte tari ale modelului:
- Gaseste de fiecare data cel putin o eticheta corecta
- Robust la adaugarea unor noi etichete
- Nu necesita antrenare suplimentara
- Posibil sa fie robust si la limba in care sunt descrise companiile :))
- Ofera mai multe etichete, nu doar una singura

## Puncte slabe ale modelului:

- Cred ca este relativ lent, dureaza in jur de secunde pentru a procesa o companie. Acest lucru poate fi rezolvat daca se aleg doar primele 5/7 din topul cosine similarity
- Ordinea in care apar etichetele in top3 poate sa nu fie intotdeauna corecta
- Din cele 3 etichete oferite, una e posibil sa nu fie foarte relevanta. Aici poate ar fi o idee buna sa verific mai multe tipuri de embeddings si modele zero-shot sau sa fac chiar putin fine-tunning.

## Scalabilitate

- Modelul este scalabil, deoarece nu necesita antrenare suplimentara si poate fi paralelizat usor.
- Pentru o viteza si mai mare se pot folosi modele de embeddings si zero shot mai rapide.
- Cu o arhitectura potrivita, modelul poate fi scalat pentru a procesa un numar mare de companii simultan.


## Alte modele incercate:

 - **Doar sentence transformer**:
   - Am incercat sa folosesc doar sentence transformer si similaritatea cosinus pentru a selecta top 3 etichete, insa rezultatele erau foarte aproapriate ca valoare si nu exista o diferentiere concreta.
- **Doar zero-shot classification**:
  - Am incercat sa folosesc doar zero-shot classification pentru a selecta top 3 etichete, insa rezultatele nu erau foarte bune, rezultatele nu erau senzationale, iar timpul de asteptare era foarte mare.
- **Gemini API**:
    - Am creat si un model decent folosit LLM-ul Gemini, rezultatele sunt ok, le voi evalua si pe acelea, dar exista sansa sa apara halucinatii si sa creeze label-uri aiurea. De asemenea, nu este scalabil, doarecere dureaza mult si interogarile costa bani.
