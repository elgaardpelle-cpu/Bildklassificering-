## Beskrivning
Detta projekt löser en klassificeringsuppgift med hjälp av Python och scikit-learn.  
Vi använder datasetet `digits` (handritade siffror 0–9) och tränar en Support Vector Machine (SVM) för att känna igen siffror.

## Arbetsgång
1. **Dataimport**  
   Datasetet `digits` laddas från scikit-learn. Varje bild är 8x8 pixlar.

2. **Förbehandling**  
   Data normaliseras med `StandardScaler` för att förbättra modellens träning.

3. **Modellval**  
   En SVM används (`gamma=0.001`, `C=10`).

4. **Träning**  
   Datan delas upp i 70% träning och 30% test. Modellen tränas på träningsdatan.

5. **Testning**  
   Modellen testas på testdatan. Noggrannhet ≈ 98%.

6. **Resultat**  
   - Klassificeringsrapport (precision, recall, F1-score).  
   - Confusion Matrix (visualisering).  
   - Cross-validation (robusthetstest).  
   - Visualisering av några testbilder med förutsägelser.  

## Resultat
- Noggrannhet: ~98%  
- Klassificeringsrapport visar hög precision för de flesta siffror, men siffran 8 är något svårare.  
- Cross-validation: genomsnittlig noggrannhet ~94.7%.  

## Körning
1. Installera beroenden:
   ```bash
   python -m pip install scikit-learn matplotlib seaborn
