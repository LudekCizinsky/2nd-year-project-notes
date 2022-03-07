#/!bin/bash

# Allow for execution certain files
chmod +x scripts.py


# initial question
while true; do
  read -p "------ Hi there, do you want to start review of week 5 exercise? [y/n] " ans
  if [[ $ans  =~ "y" ]]
  then
    break
  elif [[ $ans  =~ "n" ]]
  then
    exit 0
  fi  
done

# lecture 9
while true; do
  read -p "---- Do you want to review lecture 9 exercise? [y/n] " ans
  if [[ $ans  =~ "y" ]]
  then
     ./scripts.py "l9"
     break
  elif [[ $ans  =~ "n" ]]
  then
    break
  fi
done
