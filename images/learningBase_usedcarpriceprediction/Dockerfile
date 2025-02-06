FROM busybox

COPY ../../data/joint_data_collection.csv /learningBase/

COPY ../../data/training_data.csv /learningBase/train/

COPY ../../data/test_data.csv /learningBase/validation/

COPY images/learningbase_usedcarpriceprediction/ReadMe.md /learningBase/

# Set working directory
WORKDIR /tmp/learningBase