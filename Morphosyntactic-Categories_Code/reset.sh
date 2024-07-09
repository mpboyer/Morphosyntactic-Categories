for i in `ls ud-treebanks-v2.14`;
  do
	  \rm -r ./ud-treebanks-v2.14/"$i"/*-ud-*/;
  echo "$i";
  done;
\rm *Case_RelDep_Matches/*;
\rm *Case_Proximities/*
