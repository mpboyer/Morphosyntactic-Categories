for i in `ls ../ud-treebanks-v2.14`;
  do
	  \rm -r ../ud-treebanks-v2.14/"$i"/*-ud-*/*Case_RelDep_Matches;
    echo "$i";
  done;