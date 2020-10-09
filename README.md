# SMARTMAT Data Visualizer
## A data visualization and annotation tool for SMARTMAT data.

#### Project SMARTMAT 2.0: Activity Recognition using Smart Pressure Sensitive Mat.
##### Department of Embedded Intelligence, DFKI
##### Bo Zhou, Kumail Raza

## Installation
<code>
1. Clone the Repository<br>
2. cd smartmatvisualizer<br>
3. pip install -r requirements.txt<br>
4. put all the session files in PY/ folder<br>
</code>
<br>

## Running the tool
Run <code>python viewer.py --help </code> to get information all the supported arguments.
Required arguments are as follows:
<code>
>--session - session between 1 to 3<br>
>--bin - bin between 1 and 4<br>
>--uid - unique userID (any integer)
</code>

Optional Arguments:
<code>
>--load - (1 or 0) to load a previous session, use with UID<br>
>--sample - to start with a specific sample using sampleID in a session file<br>
>--debug - (0 or 1) to run in debug mode to display additional information on the UI as well as the terminal
</code>

Saved session json files for a user are saved in the <code>saved/</code> folder and have the following name format:
<code>
>\<uid\>\_Sess\_\<sessionID>\_Bin\_\<binID>.json<br>
>for example:    41_Sess_2_Bin_1.json </code>