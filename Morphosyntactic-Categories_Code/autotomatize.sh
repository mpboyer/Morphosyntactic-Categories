for i in `ls deep`;
  do
    \rm deep/"$i"/Sentence_Graphs.txt;
  echo "$i";
  done;