for i in `ls ud-treebanks-v2.14`;
  do
    \rm -r ./ud-treebanks-v2.14/"$i"/Sentence_Graphs/;
    \rm -r ./ud-treebanks-v2.14/"$i"/Sentence_Graphs.txt;
    \rm -r ./ud-treebanks-v2.14/"$i"/*-ud-*/;
  echo "$i";
  done;
\rm *.xlsx;