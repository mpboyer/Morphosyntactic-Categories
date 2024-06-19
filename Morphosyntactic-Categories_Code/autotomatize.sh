# python3 main.py --filename all --property obj --feature_type Case --value Nom;
# python3 main.py --filename all --property root --feature_type Case --value Acc;
# python3 main.py --filename all --property nsubj --feature_type Case --value Dat;
# python3 main.py --filename all --property case --feature_type Case --value Loc;
python3 main.py --filename all --property det --feature_type Case --value Voc;
python3 main.py --filename all --property conj --feature_type Case --value Gen;
python3 main.py --filename all --property aux --feature_type Case --value Erg;
python3 main.py --filename all --property nmod --feature_type Case --value Abs;
python3 main.py --filename all --property obl --feature_type Case --value Abl;

python3 main.py --filename all --property obj --feature_type Tense --value Past;
python3 main.py --filename all --property aux --feature_type Tense --value Pres;
python3 main.py --filename all --property amod --feature_type Tense --value Fut;
python3 main.py --filename all --property nmod --feature_type Tense --value Imp;
python3 main.py --filename all --property obl --feature_type Tense --value Pqp;

python3  main.py --filename all --property obj --feature_type Aspect --value Imp;
python3  main.py --filename all --property aux --feature_type Aspect --value Per;
python3  main.py --filename all --property obl --feature_type Aspect --value Prog;
