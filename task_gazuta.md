Iubita spor la lucrat azi.
Ca sa vezi frumos acest fisier markdown, dai click dreapta.
Si ar trebui sa ai obtiunea 
"Markdown Preview Enhanced: Open Preview to the site"
Daca nu apare inseamna ca trebuie sa instalezi ceva extensie in vscode, te prinzi tu care :).

****

## 1. Rulare initiala in runpod
vreau o data sa rulezi aplicatia fara sa aplici o transformare de orice fel, am o banuiala ca e ceva busit in pipeline, lasi pentru vreo 10 epoci si imi spui cam ce rezultate obtii, daca sunt asemanatoare cu ceea ce aveam in rularile trecute e bine, inseamna ca pipeline-ul nu este stricat, altfel va trebui sa facem debug bucata cu bucata sa vedem unde este problema.
Ca sa dezactivezi transformarile, dute in fisierul de dataset.py la linia 174, ai acolo 

```
train_ds = BraTSModalDataset(
        train_patients,
        cfg.root,
        include_random_crops=True,
        transformation=build_train_augmentations())
```

Functia build_train_augmentations creeaza pipeline-ul de transformari, poti sa il setezi pe None si asa ai dezactivat transformarile.

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

## 5 Cum paralelizam creearea setului de date in t1_out.
Se misca prea lent, ai putea sa faci un sistem multiprocess in python, care sa citeasca simultan mai multe fisiere .nii si sa creeze augmentarile pentru ele. Sa ruleze pe 4 sau 8 core-uri, ar fi mult mult mai rapid deoarece mereu recreem setul de date in runpod si e problematic cand trebuie sa astepti 20 de minute sa se creeze t1_out.  


## Important daca iti ramane timp, clasa Plotter.
Mut-o in utils, mi se pare cel mai util acolo (te-ai prins, te-ai prins).
Problema mea cu clasa asta este aproape tot main-ul, contine doar cod pentru adauga elemente in clasa Plotter,
nu e ideal, nu se mai intelege nimic din main-ul acela.
metoda plotter.update() e buna, in mod normal in codul tau
doar metoda asta ar trebui sa fie chemata.
NU in main ar trebui de initializat history, nu in main ar trebui de facut update-ul la history, si nu in main de facut printurile finale.
functia main trebuie sa aiba cat mai putine lucruri.

## sa verifc in documentatie cum au facut procesarile datelor
## sa salvez frumos codul si graficele de la fiecare experiment
