trebuie sa fiu in 3D etc... chmod +x start_runpod.sh
root@f4a694c9f22b:~/3D-Brain-Tumor-Segmentation# ./start_runpod.sh 
git clone https://github.com/ingridstoc/3D-Brain-Tumor-Segmentation.git
scp *.pth root@69.30.85.204:/root/3D-Brain-Tumor-Segmentation

schimb lr, oprimizator, scheduler, loss fct, 
adaug early stopper

t1ce, 3dunet results: === BEST MODEL ===
Best epoch: 34
Best val Dice: 0.7760
Best per-class Dice: [0.7462994706592798, 0.718262543928073, 0.8681592426198257]
Saved summary to results/t1ce_best_metrics.json

Things to try:
more class-aware crops
slightly more tumor-focused training samples
class-weighted CE term
T1ce / FLAIR input, which will help a lot more than T1 alone
If you stay with T1 only, class 1 will probably remain hard.

For each modality(t1, t2 etc), save:
best model weights
best epoch
best val mean Dice
val Dice per class

deci dupa ce fac cate o retea pt fiecare t1, t2, etc,
 urmeaza sa fac asta Compute validation Dice per class:va_pc = [c1, c2, c3], 
dupa asta Convert to weights (e.g. softmax per class):weights_c = softmax(dice_values_across_modalities) pt a face ensemble ca si cum in functie de cat valoareaza fiecare dice, sa vad cat de importanta e fiecare modalitate pt fiecare clasa, nu?

fac un tabel asa results = {
    "t1":    [0.50, 0.63, 0.52],
    "t1ce":  [0.58, 0.72, 0.55],
    "t2":    [0.52, 0.60, 0.57],
    "flair": [0.55, 0.65, 0.62],
}

sa verifc in documentatie cum au facut procesarile datelor
sa salvez frumos codul si graficele de la fiecare experiment

Iubita spor la lucrat azi.
Ca sa vezi frumos acest fisier markdown, dai click dreapta.
Si ar trebui sa ai obtiunea 
"Markdown Preview Enhanced: Open Preview to the site"
Daca nu apare inseamna ca trebuie sa instalezi ceva extensie in vscode, te prinzi tu care :).

****

## 2. Evaluare transformari
Pipeline-ul tau de transformari definit tot in dataset.py la linia 19 contine   mai multe transformari, bai ceva e ciudat la ele ca nu se comporta normal, incearca sa vezi cum se comporta cand lasi o singura transformare si le comentezi pe celelalte, scade tare acuratetea, creste? ar trebui de lasat si de rulat pentru o perioada de timp, 
parerea meste ca transformarea aceea de RandAffined strica lucrurile, dar nu sunt sigur (nu lasa pentru mai mult de 10 - 15 epoci)

## 3. Cum augmentam destept iubita.
Cat timp se ruleaza cele 2 pipeline-uri ai timp sa faci o sumedenie de alte lucruri distractive.
Vreau sa cauti toate augmentarile existente din Monai. Vreau sa faci o lista cu toate augmentarile care s-ar putea aplica in cazul nostru, de exemplu rotatiile ca RandRotate90d  sunt bune, se pot aplica, dar transformari care modifica valorile pixelilor precum GaussianNoise s-ar putea sa nu fie cele mai bune. Pe de alta parte, exista transformari care cresc contrastul intre regiuni si pot ajuta sa evidentierea tumorii, e un task dificil sa iti dai seama ce se poate aplica si ce nu. VREAU SA FACI TU LISTA SI SA INTELEGI FIECARE TRANSFORMARE IN DETALIU, vreau ca lista sa arate asa. In mod normal ti-as zice sa nu incluzi transformarile care sigur nu merg, dar as vrea sa te inveti cu ideea de a te uita in documentatie + (ceea ce faci acum se poate adauga PERFECT la sectiunea de detalii de implementare in lucrarea de licenta si sigur va ocupa o pagina sau doua, asa ca ceea ce faci poti pastra pe mai incolo)

prima transformare in monai
    - ce ai inteles tu ca face
    - cand ar fi bine de aplicat cand ar fi rau de aplicat

a doua transformare din monai
    - ce ai inteles
    - de ce ai aplica, de ce nu ai aplica transformarea

## 4 Cum testam augmentarile gasite
Trebuie sa gasim o modalitate sa testam augmentarile iubita, nu ne putem duce la ghici, e foarte riscant si putem sa ajungem sa obtinem niste rezultate de tot cacao (si nu de ala bun din Laponia). Ori o aplicatie care iti permite sa reprezenti transformarile 3d, ori un pachet in python, dar ceva trebuie gasit plus, ar merge bine de adaugat in lucrarea finala o imagine cu tumora 3d vazuta printr-un astfel de tool. Stiu ca te-ai pufines, te pupi!!!
Sarcina ta va fi sa cauti un astfel de tool si sa bagi segmentarea in tool, ca daca pui imaginea cu creierul omului nu o sa intelegem mare lucru, adica nu stiu cat de mult ajuta.

## Important daca iti ramane timp, clasa Plotter.
Mut-o in utils, mi se pare cel mai util acolo (te-ai prins, te-ai prins).
Problema mea cu clasa asta este aproape tot main-ul, contine doar cod pentru adauga elemente in clasa Plotter,
nu e ideal, nu se mai intelege nimic din main-ul acela.
metoda plotter.update() e buna, in mod normal in codul tau
doar metoda asta ar trebui sa fie chemata.
NU in main ar trebui de initializat history, nu in main ar trebui de facut update-ul la history, si nu in main de facut printurile finale.
functia main trebuie sa aiba cat mai putine lucruri.


