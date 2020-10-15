# SMARTMAT Data Visualizer
## A data visualization and annotation tool for SMARTMAT data.

#### Project SMARTMAT 2.0: Activity Recognition using Smart Pressure Sensitive Mat.
##### Department of Embedded Intelligence, DFKI
##### Bo Zhou, Kumail Raza

## Installation

1. Clone the Repository
2. <code>cd smartmatvisualizer</code>
3. <code>pip install -r requirements.txt</code>



## Running the tool
Run <code>python viewer.py --help</code> to get information all the supported arguments. Required arguments are as follows:<br>
><code>--data_path - path of the data files</code><br>
><code>--uid - unique userID (any integer)</code><br>

Optional Arguments:
><code>--load - (1 or 0) to load a previous session, use with UID</code><br>
><code>--sample - to start with a specific sample using sampleID in a session file</code><br>
><code>--debug - (0 or 1) to run in debug mode to display additional information on the UI as well as the terminal</code><br>
><code>--train - (0 or 1) To run the training session for new users</code><br>


Saved session json files for a user are saved in the <code>saved/</code> folder and have the following name format:

><code>\<uid\>.json</code><br>
><code>for example:    41.json </code>