#/!bin/bash

# Allow for execution certain files
chmod +x scripts.py


# initial question
while true; do
  read -p "------ Hi there, do you want to start review of week 6 exercise? [y/n] " ans
  if [[ $ans  =~ "y" ]]
  then
    break
  elif [[ $ans  =~ "n" ]]
  then
    exit 0
  fi  
done

# lecture 11
while true; do
  read -p "---- Do you want to review lecture 11 exercise? [y/n] " ans
  if [[ $ans  =~ "y" ]]
  then
      echo "-- Loading data, training and evaluating model..."
     ./scripts.py "l9" > lec11.log
     echo "-- See the result in lec11.log:\n>>> cat log lec11.log"
     break
  elif [[ $ans  =~ "n" ]]
  then
    break
  fi
done

# lecture 12
while true; do
  read -p "---- Do you want to review lecture 12 exercise? [y/n] " ans
  if [[ $ans  =~ "y" ]]
  then
      echo "-- Loading data, training and evaluating model..."
     ./scripts.py "l12" > lec12.log
      echo "-- See the result in lec12.log:\n>>> cat log lec12.log"
      break
  elif [[ $ans  =~ "n" ]]
  then
    break
  fi
done

