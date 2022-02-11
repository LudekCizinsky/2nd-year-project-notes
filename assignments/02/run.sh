#/!bin/bash

# initial question
while true; do
  read -p "------ Hi there, do you want to start review of week 2 exercise? [y/n] " ans
  if [[ $ans  =~ "y" ]]
  then
    break
  elif [[ $ans  =~ "n" ]]
  then
    exit 0
  fi  
done

# lecture 3
while true; do
  read -p "---- Do you want to review lecture 3 exercise? [y/n] " ans
  if [[ $ans  =~ "y" ]]
  then
     ./scripts.py "l3"
     break
  elif [[ $ans  =~ "n" ]]
  then
    break
  fi
done

# lecture 4
while true; do
  read -p "---- Do you want to review lecture 4 exercise? [y/n] " ans
  if [[ $ans  =~ "y" ]]
  then
     ./scripts.py "l4"
     break
  elif [[ $ans  =~ "n" ]]
  then
    break
  fi
done

