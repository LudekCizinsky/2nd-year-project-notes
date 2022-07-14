#/!bin/zsh

# initial question
while true; do
  read -r "ans?------ Hi there, do you want to start review of week 1 exercise? [y/n] "
  if [[ $ans  =~ "y" ]] then
    break
  elif [[ $ans  =~ "n" ]] then
    exit 0
  fi  
done

# lecture 1
while true; do
  read -r "ans?------ Do you want to review lecture 1 exercise? [y/n] "
  if [[ $ans  =~ "y" ]] then
     ./scripts.py 1 
    break
  elif [[ $ans  =~ "n" ]] then
    break
  fi  
done

# lecture 2
lecture2 () {
  echo "------ First part" 
  echo "------ searching for the first name using grep" 
  grep -n 'B-PER' da_arto.conll
  echo "------ number of names without the unique constraint"
  grep -c 'B-PER' da_arto.conll
  echo "------ number of names with the unique constraint"
  grep 'B-PER' da_arto.conll | sort | uniq | wc -l
  echo "------ making sure that we are only matching relevant column"
  awk -F\t '$2=="B-PER"' da_arto.conll
  echo "------ cleaning away the labels... "
  awk -F\t '$2=="B-PER"' da_arto.conll | awk '{print $1}'
  echo "------ Counting the number of unique names which start with a capital letter."
  awk -F\t '$2=="B-PER"' da_arto.conll | awk '{print $1}' | grep -E "^[A-Z]" | sort | uniq | wc -l
  echo "------ Counting the number of names which start with a capital letter."
  awk -F\t '$2=="B-PER"' da_arto.conll | awk '{print $1}' | grep -E "^[A-Z]" -c 
  echo "------ Second part"
  echo "------ Finding most common words in the given text" 
  sed 's/ /\n/g' pg1661.txt | sort | uniq -c | sort -r | head -4

}

while true; do
  read -r "ans?------ Do you want to review lecture 2 exercise? [y/n] "
  if [[ $ans  =~ "y" ]] then
    lecture2 
    break
  elif [[ $ans  =~ "n" ]] then
    exit 0
  fi  
done

