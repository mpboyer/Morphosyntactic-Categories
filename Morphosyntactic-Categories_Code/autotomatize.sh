Tcases=(Nom Acc Dat Loc Voc Gen Erg Abs);
for c in Tcases;
  do
    python3 main.py --filename all --property obj --feature_type Case --value "$c";
  done;
python3 main.py --filename all --property obl --feature_type Case --value Abl;

Ttense=(Past Pres Fut Imp Pqp);
for t in Ttense;
  do 
    python3 main.py --filename all --property obj --feature_type Tense --value "$t";
    done;

Taspect=(Imp Per Prog);
for a in Taspect;
  do
    python3  main.py --filename all --property obj --feature_type Aspect --value "$a";
  done;