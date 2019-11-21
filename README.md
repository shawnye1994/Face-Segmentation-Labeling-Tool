# Face-Segmentation-Labeling-Tool

## Generate Face mask by mouse painting.
## Green Brush for adding new Face region.
## Red Brush for removinng region.
## Keys:
  Crop mode: controled by 'crop_flag', default is True; For Large input image size, crop Face ROI firstly <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Caculating the IN-Polygon Pixels is time consuming for large image size.<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ESC  - exit, crop selected Rectangular region and enter into Segmentation mode <br />

  Segmentation mode: <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a     - switch to Green Brush, adding new Face region <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;r     - switch to Red Brush, removing region <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ESC   - exit and save <br />

## Mouse:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Left button down and hold: Drawing(Segmentation mode) or Create Rectangular box(Crop mode) <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Left button up(Release): Finish a bounding box selection(Crop mode, you can re-draw a box just by press and hold &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Left button again); Finish selecting a region, calculating the selected pixel area. <br />

### Tested on Win10 PC, should be cross-platform enabled
### Python 3.7.5
