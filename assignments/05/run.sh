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
      echo "-- Loading data, training and evaluating model..."
     ./scripts.py "l9" > lec9.log
     echo "-- See the result in lec9.log:\n>>> cat log lec9.log"
     break
  elif [[ $ans  =~ "n" ]]
  then
    break
  fi
done

# lecture 10
while true; do
  read -p "---- Do you want to review lecture 10 exercise? [y/n] " ans
  if [[ $ans  =~ "y" ]]
  then
      echo "-- Loading data, training and evaluating model..."
     ./scripts.py "l10" > lec10.log
      echo "-- See the result in lec10.log:\n>>> cat log lec10.log"
      break
  elif [[ $ans  =~ "n" ]]
  then
    break
  fi
done
