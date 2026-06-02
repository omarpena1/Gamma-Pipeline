From the 6/2/2026 commit onwards, this pipeline holds the functions, visualizations, and data artifacts used to produce the results of the first draft of the JINST manuscript. 

The architectural philosophy for this pipeline goes as follows:
1. Folder "/gamma/" stores `.py` files which are the computational muscles behind the different things the pipeline does: calibrating spectra, preprocessing spectra, applying statistical methods, etc...
2. The jupyter notebooks (`.ipynb`) stored in folder "/notebooks/" are flexible and light-weight wrappers which call on functions whose complicated logic is stored in the `.py` files and access the data stored in "/artifacts/".
3. Memory-heavy results, which take a lot of time to recompute (e.g. calibrated spectrum), are instead saved to fixed memory in folder "/artifacts/" (e.g. as `.npz`) so they can be reloaded quickly in jupyter notebooks. 
