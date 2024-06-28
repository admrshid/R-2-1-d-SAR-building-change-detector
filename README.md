# R-2-1-d-SAR-building-change-detector
Machine learning model for building change detection

1. Install required packages. Recommended to use separate virtual environment.
   ```python
   pip install .
2. Navigate to run
3. Produce labels by running get_label.bat. Ensure to set environment, repositroy directory, and python executale. Repository directory is "path\to\R-2-1-d-SAR-building-change-detector". Modify configs as required.
4. Files of change,no_change,{output_path}, and {output_path}_filtered will be produced after running get_label.bat. {output_path} encapsulates binary map patches of building change. {output_path}_filtered contains filtered binary map patches. change contains patches with building change. no_change contains pathces with no building change. The produced pathces serve as labels.
5. Run get_video.bat to produce series of SAR images sorted by date that spatially corresponds to the produced labels. Ensure to set python executable, environment, repository directory, data path for which SAR images are saved to, and option for segmentation. Note that code has been tailored to work with OpenInSAR, https://github.com/insar-uk/OpenInSAR.git output product.
6. Running get_video.bat will produce a file, path/to/SAR images are saved to/video_dataset.
7. Run model_run.bat to train machine learning model. Ensure to set required paths in model_run.bat. Manual modification to main file of model for further tinkering.
   
