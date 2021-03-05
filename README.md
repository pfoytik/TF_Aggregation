# TO BUILD
- docker build -t tfaggregate . 
- docker run -v $PWD:/app tfaggregate <X input file> <y input file>
  - ex. docker run -v $PWD:/app tfaggregate testX1.npy testy1.npy
  - the application will read the local file modelList.txt that contains the list of models to aggregate comma deliminated
- the docker container will start and attempt to read the necessary files in the current directory
- if successful terminal should output the accuracy score, length of test data, length of test output
