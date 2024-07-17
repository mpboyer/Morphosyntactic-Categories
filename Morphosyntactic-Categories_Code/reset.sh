for i in `ls ../ud-treebanks-v2.14`;
  do
	  \rm -r ../ud-treebanks-v2.14/"$i"/*-ud-*/AllFeatures*;
  echo "$i";
  done;
