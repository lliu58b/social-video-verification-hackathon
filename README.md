# Detect Deepfakes with Social Video Verification

*Harman Suri, Eleanor Tursman, James Tompkin*  
*Brown University*

Deepfakes can spread misinformation, defamation, andpropaganda by faking videos of public speakers. In a future where deepfakes are visually indistinguishable from real video, how will we detect them? Help build a social verification system where people group at events to capture corroborating smartphone video and verify the truth! We have built a dataset of fakes and non-fakes with tracked facial features, and we challenge you to design an algorithm to detect the faked videos. Starter code and baseline metrics included!

<img src="./images/main-fig.svg" width="70%">

## Project Familiarization:

- Visit the [social video verification webpage](http://visual.cs.brown.edu/projects/social-video-verification-webpage/).
- Watch the presentation video to gain an idea of the problem and of our proposed solution.
- Look at the 'Data' section to get an idea of the setup.
- More information about the technique is available in the [paper PDF](http://visual.cs.brown.edu/projects/social-video-verification-webpage/images/pdf-logo-over.png).

## Setup

### Environment Setup

- Python 3
- `requirements.txt` - use this to build a Python Virtual Environment for the project

### Data Download

- Follow the instructions in `./data/data.txt` to extract the pre-extracted facial landmarks.
- The input video sequences are available [here](https://repository.library.brown.edu/studio/collections/id_1006/) --- be warned, these are _large_ (5GB per person), and we recommend working with the pre-extracted facial landmarks.

Example video of the landmarks overlaid onto a person:


Example videos of real and fake sequences:

### Code

The starter code includes two files: 
- `full_sequence_exp.py` - detecting deep fakes using a whole video at once.
- `window_acc_exp.py` - detecting deep fakes using a 'sliding window' in time, to predict whether any short sequences of speech is a fake.

These files reproduce the full sequence and sliding window experiments from the paper. 

Both of these files expect the facial landmark ".mat" files to be included in the Data directory in a certain format. See the text file in the Data directory for exact naming conventions. 

_Note:_ The sliding window experiment takes a bit over an hour to run for each unique triple   (participant ID, small window size, threshold), so it might not be feasible to run over all the IDs. You may be able to translate some of the concepts from that code into your own experiments. This windowed accuracy experiment also has arguments to generate a window size vs. accuracy plot and ROC curve for a specific window size (as shown in the paper). You should include similar accuracy/ROC metrics (some graphical form preferred!) with your submission.

## Task

- Run the `full_sequence_exp.py` and generate results using our existing social deepfake detection method. You can see that it is successful about 80% of the time. 
- Can you do better?

### Possible approaches:

- Analyse the facial landmarks via their positions and motions.
- To familiarize yourself with the setup, start with a simple method based on a distance threshold on the lip distances --- the idea is to detect whether someone is 

### Submission

- Fork the repo
- Work on your method
- Commit back a `submission.md` file which includes a written description of your approach, plus includes the output plots of your performance from `full_sequence_exp.py`.
- Submit your repo URL using [this Google form](https://docs.google.com/forms/d/e/1FAIpQLSe8VrQD_oPT5Od4wpM39Xg7VwDAXIs-JxJpBPqy9ba3y3JyuQ/viewform).
- Deadline: Sunday 9th October 2020 at 3pm ET