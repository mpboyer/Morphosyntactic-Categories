for i in `ls deep`;
  do
    python3 main.py "$i" -p obj -f Case ;
  echo "$i";
  done;