# Standard imports
import copy
# Pip imports
from PIL import Image
import keras.utils
import logging
import matplotlib as mpl
import matplotlib.style as mplstyle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np
import pandas as pd
import rasterio.features
import segment_anything
import shapely
import skimage
from tqdm import tqdm
# Local imports
import segmenteverygrain

# Images larger than this will be downscaled
# 4k resolution is (2160, 4096)
IMAGE_MAX_SIZE = np.asarray((2160, 4096))

# Init logger
logger = logging.getLogger(__name__)

# Speed up rendering a little?
mplstyle.use('fast')

# Default colormap for grain visualization
DEFAULT_COLORMAP = 'tab20'

# HACK: Bypass large image restriction
Image.MAX_IMAGE_PIXELS = None

# HACK: Don't reset zoom level when pressing 'c' to create a grain
if 'c' in mpl.rcParams['keymap.back']:
    mpl.rcParams['keymap.back'].remove('c')

# HACK: Attach parent grain reference to the mpatches.Polygon class
# Makes it easy to use matplotlib "pick" event when clicking on a grain
mpatches.Polygon.grain = None


class Grain(object):
    ''' Stores data and plot representation for a single grain. '''

    def __deepcopy__(self, memo: dict):
        ''' Custom deepcopy. Avoids copying the reference image. '''
        # Create a new Grain instance
        new_grain = self.__class__(self.xy.copy())
        # Copy all attributes except for the image
        for k, v in self.__dict__.items():
            if k != 'image':
                setattr(new_grain, k, copy.deepcopy(v, memo))
        return new_grain

    def __init__(self, xy: np.ndarray, image: np.ndarray = None):
        '''
        Parameters
        ----------
        xy : list of (x, y) tuples
            Coordinates to draw this grain as a polygon.
        image : np.ndarray (optional)
            Image in which grain was detected. Used to measure color info.
        '''

        # Input
        self.image = image
        self.xy = np.array(xy)

        # Metrics
        self.data = None
        # Metrics to calculate {name: dimensionality, for unit conversion}
        self.region_props = {
            'area': 2,
            'centroid': 0,
            'major_axis_length': 1,
            'minor_axis_length': 1,
            'orientation': 0,
            'perimeter': 1,
            'max_intensity': 0,
            'mean_intensity': 0,
            'min_intensity': 0
        }

        # Display
        self.default_props = {
            'alpha': 0.6
            # facecolor is set when patch is created
        }
        self.selected_props = {
            'alpha': 1.0,
            'facecolor': 'gold'
        }
        self.axes = []
        self.patch = None
        self.selected = False

    @property
    def polygon(self) -> shapely.Polygon:
        ''' 
        Return a shapely.Polygon representing the matplotlib patch.

        Returns
        -------
        shapely.Polygon
            Polygon representing the boundaries of this grain.
        '''
        return shapely.Polygon(self.xy.T)

    def measure(self, raster: bool = False) -> pd.Series:
        '''
        Calculate grain information from image and matplotlib patch.
        Overwrites self.data.

        Parameters
        ----------
        raster : bool
            Whether to use the "old" method of measuring grain properties.
            If True, will measure all properties from a raster representation.
            If False, will prefer measurements directly from grain coords.

        Returns
        -------
        self.data : pd.Series
            Row for a DataFrame containing computed grain info.
        '''
        image = self.image
        poly = self.polygon
        props = self.region_props.keys()
        # Old method
        if raster:
            # Get rasterized shape
            rasterized = rasterio.features.rasterize(
                [poly], out_shape=image.shape[:2])
            # Calculate region properties
            data = pd.DataFrame(skimage.measure.regionprops_table(
                rasterized, intensity_image=image, properties=props))
            # Convert to row of pd.DataFrame
            if len(data):
                data = data.iloc[0]
            else:
                # TODO: Diagnose why this happens sometimes
                logger.error(f'MEASURE ERROR {pd.DataFrame(data)}')
                data = pd.Series()
        # New method, formatted to match output of old method
        else:
            data = {}
            # Size and location
            poly_metrics = measure_polygon(poly)
            if 'area' in props:
                data['area'] = poly_metrics['area']
            if 'centroid' in props:
                data['centroid-0'], data['centroid-1'] = poly_metrics['centroid']
            if 'perimeter' in props:
                data['perimeter'] = poly.length
            # Orientation and major/minor axes of best-fit ellipse
            for k, v in measure_ellipse(poly_metrics).items():
                if k in props:
                    data[k] = v
            # Color information (mean, max, min of each channel)
            if isinstance(image, np.ndarray):
                for k, v in measure_color(image, poly).items():
                    if k[:-2] in props:
                        data[k] = v
            # Convert to row of pd.DataFrame
            data = pd.Series(data)
        # Save and return results
        self.data = data
        return data

    def convert_units(self, scale: float) -> pd.Series:
        '''
        Return grian measurements scaled by a given factor.

        Parameters
        ----------
        scale : float
            Factor by which to scale grain properties.

        Returns
        -------
        data : pd.Series
            Rescaled grain measurements.
        '''
        # Skip if no measurement data exists
        if type(self.data) is type(None):
            return
        # For each type of measured value,
        data = self.data.copy()
        for k, dim in self.region_props.items():
            # If the value has any length dimensions associated with it
            if dim:
                # Scale those values by the number of length dimensions
                for col in [c for c in data.keys() if k in c]:
                    data[col] *= scale ** dim
        return data

    def erase(self):
        ''' Remove plot representation. '''
        # Remove patch
        self.patch.remove()
        # Remove axes, if drawn
        for a in self.axes:
            a.remove()

    def select(self) -> bool:
        '''
        Toggle whether grain is selected/unselected in a plot.

        Returns
        -------
        self.selected : bool
            True if this grain is now selected.
        '''
        self.selected = not self.selected
        self.patch.update(
            self.selected_props if self.selected else self.default_props)
        return self.selected

    # Drawing ----------------------------------------------------------------
    def draw_axes(self,
                  ax: mpl.axes.Axes,
                  scale: float = 1.) -> list[mpl.artist]:
        '''
        Draw centroid and major/minor axes on the provided matplotlib axes.

        Parameters
        ----------
        ax : matplotlib.Axes
            Axes instance on which to draw the grain properties.
        scale : float
            Scaling factor, useful with downscaled GrainPlots.

        Returns
        -------
        artists : { name : artist }
            Dict of Matplotlib artists, named by property that they represent.
        '''
        # Remove previously-drawn axes, if any
        for a in self.axes:
            a.remove()
        # Compute grain data if it hasn't been done already
        data = self.measure() if self.data is None else self.data
        # Centroid
        x0, y0 = data['centroid-1'] * scale, data['centroid-0'] * scale
        axes = ax.plot(x0, y0, '.k')
        # Major axis
        orientation = data['orientation']
        x = x0 - np.sin(orientation) * 0.5 * data['major_axis_length'] * scale
        y = y0 - np.cos(orientation) * 0.5 * data['major_axis_length'] * scale
        axes += ax.plot((x0, x), (y0, y), '-k', zorder=1e10)
        # Minor axis
        x = x0 + np.cos(orientation) * 0.5 * data['minor_axis_length'] * scale
        y = y0 - np.sin(orientation) * 0.5 * data['minor_axis_length'] * scale
        axes += ax.plot((x0, x), (y0, y), '-k', zorder=1e10)
        # Save and return list of drawn artists
        self.axes = axes
        return axes

    def draw_patch(self, ax: mpl.axes.Axes, scale: float = 1.) -> mpatches.Polygon:
        '''
        Draw this grain on the provided matplotlib axes and save the result.

        Parameters
        ----------
        ax : matplotlib.Axes
            Axes instance on which to draw this grain.
        scale : float
            Scaling factor, useful with downscaled GrainPlots.

        Returns
        -------
        patch
            Object representing this grain on the plot.
        '''

        # Compute grain data if it hasn't been done already
        data = self.measure() if self.data is None else self.data
        # Create patch (filled polygon)
        (patch,) = ax.fill(
            *(self.xy * scale),
            edgecolor='black',
            linewidth=2.0,
            picker=True,
            animated=True,
            **self.default_props)
        # HACK: Save reference to parent grain within the patch itself
        patch.grain = self
        # Save assigned color (for select/unselect)
        self.default_props['facecolor'] = patch.get_facecolor()
        # Save and return reference to drawn patch
        self.patch = patch
        return patch


class GrainPlot(object):
    ''' 
    Interactive plot to create, delete, and merge grains.
    
    Mouse Controls:
    - Left-click on mask-free area: Create grain immediately (auto-create)
    - Left-click on existing grain: Select/unselect grain
    - Right-click on mask-free area: Add background prompt and create grain
    - Alt + Left-click: Add foreground prompt (hold Alt for multiple prompts)
    - Alt + Right-click: Add background prompt (hold Alt for multiple prompts)
    - Shift+Left-click: Show grain measurement info (alternative to middle-click)
    - Shift+Left-drag: Draw scale bar line (red line shows the scale reference)
    - Middle-click: Show grain measurement info
    
    Keyboard Controls:
    - Alt (hold): Enable multiple prompt mode (click to place prompts, release Alt to create)
    - M: Merge selected grains
    - Shift (hold): Enable scale bar drawing mode
    - Ctrl (hold): Hide all grain segmentations temporarily
    - C: Create grain from existing prompts (alternative to auto-create)
    - D/Delete: Delete selected grains
    - Z: Undo last created grain
    - Escape: Clear all selections and prompts
    '''

    def __init__(self,
                 grains: list = [],
                 image: np.ndarray = None,
                 predictor: segment_anything.SamPredictor = None,
                 blit: bool = True,
                 px_per_m: float = 1.,       # px/m
                 scale_m: float = 1.,        # m
                 minspan: int = 10,          # px
                 image_alpha: float = 1.,
                 image_max_size: tuple[float, float] = IMAGE_MAX_SIZE,
                 color_palette: str = DEFAULT_COLORMAP,  # Matplotlib colormap name
                 color_by: str = None,       # Property to color grains by ('major_axis_length', 'minor_axis_length', 'area', etc.)
                 **kwargs):
        '''
        Parameters
        ----------
        grains : list
            List of grains with xy data to plot over the backround image.
        image : np.ndarray
            Image under analysis, displayed behind identified grains.
        predictor : segment_anything.SamPredictor
            SAM predictor used to create new grains.
        blit : bool, default True
            Whether to use blitting (much faster, potentially buggy).
        px_per_m : float, default 1.0
            Pixels per meter for unit conversion.
        scale_m : float, default 1.0
            Length of any scale bar represented in the image, in meters.
        minspan : int, default 10
            Minimum size for box selector tool.
        image_alpha : float, default 1.0
            Alpha value for background image, passed to imshow().
        image_max_size : (y, x)
            Images larger than this will be downscaled for display.
            Grain creation and measurement will still use the full image.
        color_palette : str, default 'tab20'
            Name of any valid matplotlib colormap for grain visualization 
            (e.g., 'tab20', 'Set3', 'Paired', 'viridis', 'magma', 'plasma', 
            'twilight', 'hsv', etc.). Colors are randomly sampled from the 
            colormap for visual variety unless color_by is specified.
            See matplotlib's colormap reference for all available options.
        color_by : str, optional
            Property to use for coloring grains. If specified, grains will be
            colored based on the values of this property using the colormap.
            Common options: 'major_axis_length', 'minor_axis_length', 'area',
            'perimeter', 'orientation'. If None (default), colors are randomly
            sampled from the colormap.
        kwargs : dict
            Keyword arguments to pass to plt.figure().
        '''
        logger.info('Creating GrainPlot...')
        
        # Store color palette and color_by selection for later use
        self.color_palette = color_palette
        self.color_by = color_by

        # Events
        self.cids = []
        self.events = {
            'button_press_event': self.onclick,
            'button_release_event': self.onclickup,
            'motion_notify_event': self.onmotion,
            'draw_event': self.ondraw,
            'key_press_event': self.onkey,
            'key_release_event': self.onkeyup,
            'pick_event': self.onpick}

        # Interaction history
        self.ctrl_down = False
        self.alt_down = False  # Track if 'Alt' key is held for multiple prompts
        self.last_pick = (0, 0)
        self.points = []
        self.point_labels = []
        self.created_grains = []
        self.selected_grains = []

        # Plot
        self.blit = blit
        self.fig = plt.figure(**kwargs)
        self.canvas = self.fig.canvas
        self.ax = self.fig.add_subplot(aspect='equal', xticks=[], yticks=[])

        # Background image
        self.predictor = predictor
        self.image = image
        self.display_image = image
        self.scale = 1.
        if isinstance(image, np.ndarray):
            # Downscale image if needed
            max_size = np.asarray(image_max_size)
            if image.shape[0] > max_size[0] or image.shape[1] > max_size[1]:
                logger.info('Downscaling large image for display...')
                self.scale = np.min(max_size / image.shape[:2])
                self.display_image = skimage.transform.rescale(
                    image, self.scale, anti_aliasing=False, channel_axis=2)
                logger.info(f'Downscaled image to {self.scale} of original.')
            # Show image
            self.ax.imshow(self.display_image, alpha=image_alpha)
            self.ax.autoscale(enable=False)
        self.fig.tight_layout(pad=0)

        # Interactive toolbar: inject clear_all before any zoom/pan changes
        # Avoids manual errors and bugs with blitting
        toolbar = self.canvas.toolbar
        if hasattr(toolbar, '_update_view'):
            toolbar._update_view = self._clear_before(toolbar._update_view)
            toolbar.release_pan = self._clear_before(toolbar.release_pan)
            toolbar.release_zoom = self._clear_before(toolbar.release_zoom)

        # Box selector
        self.minspan = minspan
        self.box = np.zeros(4, dtype=int)
        self.box_selector = mwidgets.RectangleSelector(
            self.ax,
            onselect=lambda *args: None,  # Don't do anything on selection
            minspanx=minspan,             # Minimum selection size
            minspany=minspan,
            useblit=True,                 # Always try to use blitting
            props={
                'facecolor': 'gold',
                'edgecolor': 'black',
                'alpha': 0.2,
                'fill': True},
            spancoords='pixels',
            button=[1],                   # Left mouse button only
            interactive=True,
            state_modifier_keys={})       # Disable shift/ctrl modifiers
        self.box_selector.set_active(False)
        # Replace RectangleSelector update methods to avoid redundant blitting
        if blit:
            self.box_selector.update = self.update
            self.box_selector.update_background = lambda *args: None

        # Scale bar (drawn as a line)
        self.px_per_m = px_per_m
        self.scale_m = scale_m
        self.scale_line = None  # Will hold the matplotlib Line2D object
        self.scale_drawing = False  # Track if currently drawing scale line
        self.scale_start = None  # Starting point of scale line

        # Info box
        self.info = self.ax.annotate(
            '',
            xy=(0, 0),
            xytext=(0, 0),
            textcoords='offset points',
            ha='center',
            va='center',
            bbox={'boxstyle': 'round', 'fc': 'w'},
            animated=blit)
        self.info_grain = None
        self.info_grain_candidate = None
        
        # Generate colors from the selected colormap
        if color_by:
            logger.info(f'Generating colors from "{color_palette}" colormap based on "{color_by}".')
            grain_colors = self._generate_colors_by_property(grains, color_palette, color_by)
        else:
            logger.info(f'Generating colors from "{color_palette}" colormap.')
            grain_colors = self._generate_colors(len(grains), color_palette)
        
        # Draw grains and initialize plot
        logger.info('Drawing grains.')
        self.grains = grains
        for i, grain in enumerate(tqdm(grains, desc='Measuring and drawing grains')):
            grain.image = image
            grain.measure()
            # Set the grain's color before drawing
            grain.default_props['facecolor'] = grain_colors[i]
            grain.draw_patch(self.ax, self.scale)
        if blit:
            self.artists = [self.info,
                            *self.box_selector.artists]
            self.canvas.draw()
        
        # Add keyboard shortcuts to the window title
        shortcuts_title = (
            'GrainPlot | Click=Auto-create, Alt+Click=Multi-prompt, Shift+Drag=Scale, '
            'D=Delete, M=Merge, Z=Undo, Esc=Clear, Ctrl=Hide'
        )
        self.fig.canvas.manager.set_window_title(shortcuts_title)
        
        logger.info('GrainPlot created!')

    # Color generation -------------------------------------------------------
    def _generate_colors(self, n_grains: int, cmap_name: str) -> list:
        '''
        Generate colors for grains from the specified matplotlib colormap.
        
        Parameters
        ----------
        n_grains : int
            Number of colors to generate
        cmap_name : str
            Name of the matplotlib colormap to use (e.g., 'tab20', 'Set3', 
            'viridis', 'magma', etc.). Any valid matplotlib colormap name 
            can be used.
            
        Returns
        -------
        colors : list
            List of RGBA color tuples
        '''
        try:
            # Get the colormap
            cmap = plt.cm.get_cmap(cmap_name)
            # Generate random indices for variety (similar to plot_image_w_colorful_grains)
            color_indices = np.random.randint(0, cmap.N, n_grains)
            colors = [cmap(i) for i in color_indices]
            return colors
        except ValueError:
            # If colormap name is invalid, fall back to default
            logger.warning(f'Unknown colormap "{cmap_name}", using "{DEFAULT_COLORMAP}"')
            cmap = plt.cm.get_cmap(DEFAULT_COLORMAP)
            color_indices = np.random.randint(0, cmap.N, n_grains)
            colors = [cmap(i) for i in color_indices]
            return colors
    
    def _generate_colors_by_property(self, grains: list, cmap_name: str, property_name: str) -> list:
        '''
        Generate colors for grains based on a property value using a colormap.
        
        Parameters
        ----------
        grains : list
            List of Grain objects
        cmap_name : str
            Name of the matplotlib colormap to use
        property_name : str
            Name of the grain property to use for coloring (e.g., 'major_axis_length')
            
        Returns
        -------
        colors : list
            List of RGBA color tuples mapped to property values
        '''
        try:
            # Get the colormap
            cmap = plt.cm.get_cmap(cmap_name)
            
            # Extract property values from grains
            # First ensure all grains are measured
            property_values = []
            for grain in grains:
                if grain.data is None:
                    grain.measure()
                
                # Get the property value
                if property_name in grain.data:
                    property_values.append(grain.data[property_name])
                else:
                    logger.warning(f'Property "{property_name}" not found in grain data, using 0')
                    property_values.append(0)
            
            property_values = np.array(property_values)
            
            # Normalize values to [0, 1] range
            if len(property_values) > 0 and np.ptp(property_values) > 0:
                normalized_values = (property_values - np.min(property_values)) / np.ptp(property_values)
            else:
                # If all values are the same, use middle of colormap
                normalized_values = np.full(len(property_values), 0.5)
            
            # Map normalized values to colors
            colors = [cmap(val) for val in normalized_values]
            
            logger.info(f'Color range: {property_name} from {np.min(property_values):.3f} to {np.max(property_values):.3f}')
            
            return colors
            
        except ValueError as e:
            # If colormap name is invalid, fall back to random colors
            logger.warning(f'Error generating colors by property: {e}, using random colors')
            return self._generate_colors(len(grains), DEFAULT_COLORMAP)

    # Display helpers --------------------------------------------------------
    def _clear_before(self, f: object) -> object:
        ''' 
        Wrap a function to call self.clear_all() before it.

        Parameters
        ----------
        f : function
            The function or method to wrap.

        Returns
        -------
        newf : function
            The given function, wrapped to call self.clear_all() first.
        '''
        def newf(*args, **kwargs):
            if self.blit:
                self.clear_all()
            return f(*args, **kwargs)
        return newf

    def update(self):
        ''' Blit background image and draw animated artists. '''
        # If not blitting, just request a redraw and return
        # Apparently necessary if plot shown twice
        if not self.blit:
            self.canvas.draw_idle()
            return
        # Reset background from image
        self.canvas.restore_region(self.background)
        # Draw animated artists
        # TODO: More efficient to maintain this list elsewhere?
        info_grain = self.info_grain
        info_patch = [info_grain.patch] if info_grain is not None else []
        scale_line = [self.scale_line] if self.scale_line is not None else []
        artists = ([g.patch for g in self.selected_grains]
                   + info_patch
                   + self.points
                   + scale_line
                   + self.artists)
        for a in artists:
            self.ax.draw_artist(a)
        # Push to canvas
        self.canvas.blit(self.ax.bbox)

    def draw_axes(self):
        ''' 
        Draw the major and minor axes on each grain patch.
        
        This method collects all axis data first, then plots them all at once
        on top of the grain masks to ensure proper z-ordering.
        
        Note: Call this after deactivate() to draw axes on top of grain masks.
        '''
        # First, make sure all grain patches are non-animated so they render normally
        for grain in self.grains:
            grain.patch.set_animated(False)
        
        # Force a full canvas draw to render all grain patches
        self.canvas.draw()
        
        # Collect all centroid and axis line data
        centroids_x = []
        centroids_y = []
        major_lines = []
        minor_lines = []
        
        for grain in self.grains:
            # Compute grain data if it hasn't been done already
            data = grain.measure() if grain.data is None else grain.data
            
            # Centroid coordinates
            x0 = data['centroid-1'] * self.scale
            y0 = data['centroid-0'] * self.scale
            centroids_x.append(x0)
            centroids_y.append(y0)
            
            # Major axis line
            orientation = data['orientation']
            x_major = x0 - np.sin(orientation) * 0.5 * data['major_axis_length'] * self.scale
            y_major = y0 - np.cos(orientation) * 0.5 * data['major_axis_length'] * self.scale
            major_lines.append([(x0, y0), (x_major, y_major)])
            
            # Minor axis line
            x_minor = x0 + np.cos(orientation) * 0.5 * data['minor_axis_length'] * self.scale
            y_minor = y0 - np.sin(orientation) * 0.5 * data['minor_axis_length'] * self.scale
            minor_lines.append([(x0, y0), (x_minor, y_minor)])
        
        # Plot all centroids at once on top (high zorder)
        self.ax.plot(centroids_x, centroids_y, '.k', markersize=3, zorder=1000)
        
        # Plot all major axes
        for line in major_lines:
            self.ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 
                        '-k', linewidth=1, zorder=1000)
        
        # Plot all minor axes
        for line in minor_lines:
            self.ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 
                        '-k', linewidth=1, zorder=1000)
        
        # Force another draw to render the axes
        self.canvas.draw()
    
    def get_grains(self) -> list:
        ''' 
        Get the current list of grains.
        
        This returns the live grain list that reflects all interactive edits
        (created, deleted, merged grains). You can use this to save the current
        state without needing to write to disk first.
        
        Returns
        -------
        grains : list
            Current list of Grain objects, including all edits.
        '''
        return self.grains

    # Measurements -----------------------------------------------------------
    def draw_scale_line(self, start, end):
        '''
        Draw a scale bar line and calculate the scale based on its length.

        Parameters
        ----------
        start: tuple
            Starting point (x, y) of the scale line.
        end: tuple
            Ending point (x, y) of the scale line.
        '''
        # Calculate line length in pixels
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        px = np.sqrt(dx * dx + dy * dy)
        
        # Verify that the line is big enough
        if px < self.minspan:
            logger.warning(f'Scale line too short ({px:.1f} < {self.minspan})')
            return
        
        # Remove old scale line if it exists
        if self.scale_line is not None:
            self.scale_line.remove()
        
        # Draw a red line to mark the scale bar
        self.scale_line = self.ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            'r-',  # Red solid line
            linewidth=3,
            animated=self.blit,
            zorder=1000  # Draw on top of everything
        )[0]
        
        # Convert to pixels per meter using a known scale bar length
        px_per_m = px / self.scale_m / self.scale
        self.px_per_m = px_per_m
        logger.info(f'Scale set to {px_per_m:.2f} pixels per meter.')
        
        # Update the info box using the new units
        self.update_info(self.info_grain)
        self.update()

    def clear_info(self):
        ''' Clear the grain info box. '''
        self.update_info(None)

    def update_info(self, grain: Grain = None):
        ''' Update info box for specified grain. '''
        # Restore proper color to previous info_grain
        old_grain = self.info_grain
        if old_grain is not None:
            if old_grain.selected:
                props = old_grain.selected_props
            else:
                props = old_grain.default_props
            old_grain.patch.update(props)
        # Hide info box and return if:
        # - No new grain given
        # - New grain is the same as the old grain
        if (grain is None
                or (grain is old_grain
                    and grain is self.info_grain_candidate)):
            self.info_grain = None
            self.info_grain_candidate = None
            self.info.set_visible(False)
            return
        # Update saved info_grain
        self.info_grain = grain
        # Highlight new info_grain
        grain.patch.set_facecolor('blue')
        # Determine info box position offset based on grain's position
        ext = grain.patch.get_extents()
        img_x, img_y = self.canvas.get_width_height()
        x = -0.1 if (ext.x1 + ext.x0) / img_x > 1 else 1.1
        y = -0.1 if (ext.y1 + ext.y0) / img_y > 1 else 1.1
        # Extra offset to avoid covering up small grains
        if abs(ext.y1 - ext.y0) < img_y / 20:
            if y <= 0:
                y -= 1.4
            else:
                y += 1.4
        # Update position
        info = self.info
        info.xy = (x, y)
        info.xycoords = grain.patch
        # Update text, converting units if needed
        if self.px_per_m == 1.:
            data = grain.data
            text = (f"Major: {data['major_axis_length']:.0f} px\n"
                    f"Minor: {data['minor_axis_length']:.0f} px\n"
                    f"Area: {data['area']:.0f} px")
        else:
            data = grain.convert_units(1 / self.px_per_m)
            text = (f"Major: {data['major_axis_length']:.3} m\n"
                    f"Minor: {data['minor_axis_length']:.3} m\n"
                    f"Area: {data['area']:.3} m$^2$")
        info.set_text(text)
        # Show info box
        self.info.set_visible(True)

    # Selection helpers ------------------------------------------------------
    def toggle_box(self, active: bool = None) -> bool:
        ''' 
        Activate/deactivate selection box.

        Parameters
        ----------
        active : bool
            Whether to activate (True) or deactivate (False) the selection box.
            If not provided, will toggle current active status.

        Returns
        -------
        active : bool
            Updated active status of the selection box. '''
        # Clear selected grains; selection box means we're making a new one
        self.clear_grains()
        # Get new active state
        box = self.box_selector
        active = active or not box.get_active()
        # Set visual properties
        alpha = 0.4 if active else 0.2
        box.set_handle_props(alpha=alpha, visible=active)
        box.set_props(alpha=alpha)
        box.set_active(active)
        return active

    def clear_box(self):
        ''' Alias for self.box_selector.clear(). '''
        self.box_selector.clear()

    def set_point(self, xy: tuple[int, int], foreground: bool = True) -> mpatches.Circle:
        ''' 
        Set point prompt, either foreground or background.

        Parameters
        ----------
        xy : (x: int, y: int)
            Coordinates of newly-requested point.
        foreground : bool
            Whether point represents a foreground (True) or background prompt.

        Returns
        -------
        new_point : mpatches.Circle
        '''
        new_point = mpatches.Circle(
            xy,
            radius=5,
            color='lime' if foreground else 'red',
            animated=self.blit)
        self.ax.add_patch(new_point)
        self.points.append(new_point)
        self.point_labels.append(foreground)
        return new_point

    def clear_points(self):
        ''' Clear all prompt points. '''
        for point in self.points:
            point.remove()
        self.points = []
        self.point_labels = []

    def clear_grains(self):
        ''' Unselect all selected grains. '''
        for grain in self.selected_grains:
            grain.select()
        self.selected_grains = []
        self.update_info()

    def clear_all(self):
        ''' Clear prompts, unselect all grains, and hide the info box. '''
        self.clear_box()
        self.clear_grains()
        self.clear_info()
        self.clear_points()

    # Manage grains ----------------------------------------------------------
    def create_grain(self) -> Grain:
        ''' 
        Attempt to detect a grain based on given prompts.

        Returns
        -------
        new_grain : Grain
            Grain detected using previously-input prompts.
        '''
        # Interpret point prompts
        if len(self.points):
            points = [
                np.asarray(p.get_center()) / self.scale for p in self.points]
            point_labels = self.point_labels
        else:
            points = None
            point_labels = None
        # Interpret box prompt
        if self.box_selector._selection_completed:
            xmin, xmax, ymin, ymax = np.asarray(
                self.box_selector.extents) / self.scale
            box = np.asarray((xmin, ymin, xmax, ymax))
        else:
            # Return if we haven't provided any prompts
            if points is None:
                return
            box = None
        # Use prompts to find a grain
        coords = predict_from_prompts(
            predictor=self.predictor,
            box=box,
            points=points,
            point_labels=point_labels)
        
        # Check if SAM failed to produce a valid mask
        if coords[0] is None or coords[1] is None:
            logger.warning('SAM failed to produce a valid mask from prompts')
            # Clear prompts and return without creating a grain
            self.clear_all()
            self.update()
            return None
        
        # Scale and record new grain (on plot, data, and undo list)
        new_grain = Grain(coords, self.image)
        new_grain.draw_patch(self.ax, self.scale)
        self.grains.append(new_grain)
        self.created_grains.append(new_grain)
        # Clear prompts and update background
        self.clear_all()
        if self.blit:
            self.canvas.draw()
        return new_grain

    def delete_grains(self):
        ''' Delete all selected grains. '''
        # If no individual grains are selected...
        if len(self.selected_grains) < 1:
            # Use any grains that are wholly contained in the box selector
            if self.box_selector._selection_completed:
                xmin, xmax, ymin, ymax = np.asarray(
                    self.box_selector.extents) / self.scale
                box = shapely.box(xmin, ymin, xmax, ymax)
                self.selected_grains = [
                    g for g in self.grains if box.contains(g.polygon)]
            # Otherwise, exit since no grains have been indicated for deletion
            else:
                return
        # Remove selected grains from plot, data, and undo list
        for grain in self.selected_grains:
            grain.erase()
            self.grains.remove(grain)
            if grain in self.created_grains:
                self.created_grains.remove(grain)
        # Clear any prompts (assumed accidental) and update background
        self.clear_all()
        if self.blit:
            self.canvas.draw()

    def hide_grains(self, hide: bool = True):
        ''' 
        Hide or unhide all grains.

        Parameters
        ----------
        hide : bool
            Whether to hide (True) or unhide (False) all grains.

        Returns
        -------
        hide : bool
            New hidden status.
        '''
        # Hide other elements if needed
        if hide:
            self.clear_box()
            self.clear_info()
            self.clear_points()
        # Show/hide all grains
        for grain in self.grains:
            grain.patch.set_visible(not hide)
        # Update background
        if self.blit:
            self.canvas.draw()
        # Set selected grains to default color when hidden,
        # or restore selected color when unhidden.
        # Avoids drawing wrong color into background on unhide.
        for grain in self.selected_grains:
            grain.select()
        return hide

    def merge_grains(self) -> Grain:
        ''' 
        Attempt to merge all selected grains.

        Returns
        -------
        new_grain : Grain
            Merged grain, if merge is successful. None otherwise.
        '''
        # Verify there are at least two grains selected to merge
        if len(self.selected_grains) < 2:
            return
        # Find vertices of merged grains using Shapely
        poly = shapely.unary_union(
            [g.polygon for g in self.selected_grains])
        # Verify grains actually overlap, otherwise reject selections
        if isinstance(poly, shapely.MultiPolygon):
            self.clear_grains()
            return
        # Make new merged grain
        new_grain = Grain(poly.exterior.xy, self.image)
        new_grain.draw_patch(self.ax, self.scale)
        self.grains.append(new_grain)
        self.created_grains.append(new_grain)
        # Delete old constituent grains (since they are still selected)
        self.delete_grains()
        return new_grain

    def undo_grain(self):
        ''' Remove latest created grain. '''
        # TODO: Also allow undoing grain deletions
        # Verify that there is a grain to undo
        if len(self.created_grains) < 1:
            return
        # Select and remove latest grain
        self.clear_all()
        self.selected_grains = [self.created_grains[-1]]
        self.delete_grains()

    # Events -----------------------------------------------------------------
    def onclick(self, event: mpl.backend_bases.MouseEvent):
        '''
        Handle clicking anywhere on plot.
        Places foreground or background prompts for grain creation.
        Shift + Left drag draws a scale bar line.

        Parameters
        ----------
        event : MouseEvent
            Event details
        '''
        # If shift is held and left button, start drawing scale line
        if event.key == 'shift' and event.button == 1 and event.inaxes == self.ax:
            # Deactivate box selector to prevent rectangle from being drawn
            self.box_selector.set_active(False)
            self.scale_drawing = True
            self.scale_start = (event.xdata, event.ydata)
            return
        
        # Ignore click if:
        # - It's in a different axes
        # - It's a double-click
        # - Toolbar interaction is happening (pan/zoom)
        # - Ctrl is held (hiding grains)
        # - Box selection is active
        # - Grains are selected
        # - It was already handled by onpick
        if (event.inaxes != self.ax
                or event.dblclick
                or self.canvas.toolbar.mode != ''
                or self.ctrl_down
                or self.box_selector.get_active()
                or len(self.selected_grains) > 0
                or self.last_pick == (round(event.xdata), round(event.ydata))):
            return
        # Left click: Foreground prompt
        button = event.button
        coords = (event.xdata, event.ydata)
        if button == 1:
            self.set_point(coords, True)
            # If 'Alt' key is NOT held, auto-create grain immediately
            if not self.alt_down:
                self.create_grain()
                return  # create_grain() calls update()
        # Right click: Background prompt
        elif button == 3:
            self.set_point(coords, False)
            # If 'Alt' key is NOT held, auto-create grain immediately
            if not self.alt_down:
                self.create_grain()
                return  # create_grain() calls update()
        # Only update display if something happened
        else:
            # Forget any grain previously picked with scrollwheel click
            self.info_grain_candidate = None
            return
        self.update()

    def onclickup(self, event: mpl.backend_bases.MouseEvent):
        '''
        Handle click release events anywhere on plot.
        Completes scale bar line drawing if in progress.
        Displays info about any grain indicated with middle-click or Shift+left-click.

        Parameters
        ----------
        event : MouseEvent
            Event details
        '''
        # Complete scale line drawing if in progress
        if self.scale_drawing and event.button == 1 and event.inaxes == self.ax:
            self.scale_drawing = False
            scale_end = (event.xdata, event.ydata)
            self.draw_scale_line(self.scale_start, scale_end)
            self.scale_start = None
            return
        
        # Respond to middle-click OR shift+left-click release without dragging
        is_info_click = (event.button == 2) or (event.button == 1 and event.key == 'shift')
        
        if (not is_info_click
                or self.last_pick != (round(event.xdata), round(event.ydata))):
            return
        # Update info box and display
        self.update_info(self.info_grain_candidate)
        self.update()

    def onmotion(self, event: mpl.backend_bases.MouseEvent):
        '''
        Handle mouse motion events.
        Updates the scale line preview while dragging.

        Parameters
        ----------
        event : MouseEvent
            Event details
        '''
        # Update scale line preview if currently drawing
        if self.scale_drawing and event.inaxes == self.ax:
            # Remove old preview line if it exists
            if self.scale_line is not None:
                self.scale_line.remove()
            
            # Draw preview line from start to current position
            self.scale_line = self.ax.plot(
                [self.scale_start[0], event.xdata],
                [self.scale_start[1], event.ydata],
                'r-',  # Red solid line
                linewidth=3,
                animated=self.blit,
                zorder=1000  # Draw on top of everything
            )[0]
            
            # Update display
            self.update()

    def ondraw(self, event: mpl.backend_bases.DrawEvent):
        ''' 
        Update saved background image whenever a full redraw is triggered.

        Parameters
        ----------
        event : DrawEvent
            Event details
        '''
        if self.blit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def onkey(self, event: mpl.backend_bases.KeyEvent):
        '''
        Handle key presses as follows:
        c: Create a grain from existing prompts (alternative to auto-create).
        d: Delete a selected grain.
        alt (hold): Allow placing multiple prompts before creating grain.
        m: Merge selected grains.
        z: Undo the most recently created grain.
        control (hold): Temporarily hide all grains.
        escape: Remove all selections, prompts, and info.
        shift: Activate scale bar selector and grain selection box.

        Parameters
        ----------
        event : KeyEvent
            Event details
        '''
        # Ignore keypress if:
        # - It's from another axes
        # - Grains are hidden
        # - Box selection is active
        if (not event.inaxes == self.ax
                or self.ctrl_down
                or self.box_selector.get_active()):
            return
        # Handle keypress appropriately
        key = event.key
        if key == 'c':
            self.create_grain()
        elif key == 'd' or key == 'delete':
            self.delete_grains()
        elif key == 'alt':
            # Set flag to allow multiple prompts
            self.alt_down = True
            return  # Don't update, just set the flag
        elif key == 'm':
            # Merge selected grains
            self.merge_grains()
        elif key == 'z':
            self.undo_grain()
        elif key == 'control':
            self.ctrl_down = True
            self.hide_grains()
        elif key == 'escape':
            self.clear_all()
        elif key == 'shift':
            # Don't activate box selector on shift press - it will be handled
            # in onclick if needed. This allows shift+drag for scale line drawing.
            pass
        # Only update display if something happened
        else:
            return
        self.update()

    def onkeyup(self, event: mpl.backend_bases.KeyEvent):
        ''' 
        Handle key releases.
        Unhides hidden grains, creates grain from multiple prompts, or deactivates selectors.

        Parameters
        ----------
        event : KeyEvent
            Event details.
        '''
        # Ignore key release from other axes
        if not event.inaxes == self.ax:
            return
        # Handle release appropriately
        key = event.key
        if key == 'control':
            self.ctrl_down = False
            self.hide_grains(False)
        elif key == 'alt':
            # Release 'Alt' key: create grain from accumulated prompts if any exist
            self.alt_down = False
            if len(self.points) > 0:
                self.create_grain()
                return  # create_grain() calls update()
        elif key == 'shift':
            # Nothing to do on shift release - scale line drawing is handled in onclickup
            pass
        # Only update display if something happened
        else:
            return
        self.update()

    def onpick(self, event: mpl.backend_bases.PickEvent):
        '''
        Handle clicking on an existing grain.
        Left-click: Select/unselect the grain.
        Middle-click OR Shift+Left-click: Show measurement info for indicated grain.
        Right-click: Ignored (passes through to onclick for background prompts).

        Parameters
        ----------
        event : PickEvent
            Event details
        '''
        # Ignore the pick event if:
        # - Double-clicking
        # - Navigating via the toolbar (pan/zoom)
        # - Grains are hidden (ctrl held)
        # - Box selection is active
        mouseevent = event.mouseevent
        if (mouseevent.dblclick
                or self.canvas.toolbar.mode != ''
                or self.ctrl_down
                or self.box_selector.get_active()):
            return
        
        # Get button type
        button = mouseevent.button
        
        # Right-click: Don't handle here - let it pass through to onclick for background prompts
        if button == 3:
            return
        
        # Save click location for reference by onclick / onclickup
        self.last_pick = (round(mouseevent.xdata), round(mouseevent.ydata))
        
        # Left-click: Select grain if no point prompts exist
        if button == 1 and len(self.points) == 0:
            # Shift+Left-click: Show info (alternative to middle-click)
            if mouseevent.key == 'shift':
                self.info_grain_candidate = event.artist.grain
            else:
                # Regular left-click: Add/remove selected grain to/from selection list
                grain = event.artist.grain
                if grain.select():
                    self.selected_grains.append(grain)
                else:
                    self.selected_grains.remove(grain)
        # Middle-click: Save grain to show info if click is not dragged
        elif button == 2:
            self.info_grain_candidate = event.artist.grain
        # Only display if something happened
        else:
            return
        self.update()

    def activate(self):
        ''' Enable interactive features. '''
        cids, canvas = self.cids, self.canvas
        for event, handler in self.events.items():
            cids.append(canvas.mpl_connect(event, handler))

    def deactivate(self):
        ''' Disable interactive features. '''
        canvas = self.canvas
        for cid in self.cids:
            canvas.mpl_disconnect(cid)
        self.cids = []

    # Output -----------------------------------------------------------------
    def savefig(self, fn: str):
        ''' 
        Save figure to disk. 

        Parameters
        ----------
        fn : str
            Filename for output image. File type is determined by extension.
        '''
        self.fig.savefig(fn, bbox_inches='tight', pad_inches=0)

def predict_from_prompts(predictor, box=None, points=None, point_labels=None):
    """
    Perform a point-prompt-based segmentation using the SAM model. 

    Parameters
    ----------
        predictor
            SAM predictor
        image: numpy.ndarray
            Input image
        box: list, optional
            Selection box coordinates, as [xmin, ymin, xmax, ymax]
        points: list, optional
            Prompt coordinates, as [(x1, y1), (x2, y2)...]
        point_labels: list, optional
            Bool values for each point, True if inside the object

    Returns
    -------
        contour: list
            Points along resulting contour, as [(x1, y1), (x2, y2)...]
    """
    # Format input
    if points is not None:
        points = np.array(points)
        point_labels = np.array(point_labels)
    if box is not None:
        box = np.array(box)[None, :]
    # Do segmentation
    masks, _, _ = predictor.predict(
        point_coords=points,
        point_labels=point_labels,
        box=box,
        # TODO: Is multimask_output=True better when using only one prompt?
        multimask_output=False
    )
    
    # Get mask and handle edge grains using padding technique
    mask = masks[0].astype(bool)
    
    # Check if mask touches any edge of the image
    touches_edge = (
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )
    
    if touches_edge:
        # Pad the mask with 1 pixel border to allow proper contour detection
        # This is the same technique used in segmenteverygrain.one_point_prompt
        mask = np.pad(mask, 1, mode="constant")
        contours = skimage.measure.find_contours(mask.astype(float), 0.5)
        if len(contours) == 0:
            # No contours found - return None to indicate failure
            return None, None
        sx, sy = contours[0][:, 1], contours[0][:, 0]
        # Adjust coordinates back to account for padding
        if np.any(mask[1, :]):
            sy = sy - 1
        if np.any(mask[:, 1]):
            sx = sx - 1
    else:
        # Standard contour detection for non-edge grains
        contours = skimage.measure.find_contours(mask.astype(float), 0.5)
        if len(contours) == 0:
            # No contours found - return None to indicate failure
            return None, None
        sx, sy = contours[0][:, 1], contours[0][:, 0]
    
    return sx, sy


# Input/output ---------------------------------------------------------------

def load_image(fn: str) -> np.ndarray:
    ''' 
    Load an image from disk as a numpy array.

    Parameters
    ----------
    fn : str
        Filename for image to load.

    Returns
    -------
    np.ndarray
        Memory representation of loaded image.
    '''
    return np.array(keras.utils.load_img(fn))


def polygons_to_grains(polygons: list, image: np.ndarray = None) -> list:
    ''' 
    Construct grains from a list of polygons defining grain boundaries.

    Parameters
    ----------
    polygons : list of shapely.Polygon
        Polygons defining grain boundaries.

    Returns
    -------
    list
        Grain objects created from provided polygons.
    '''
    grains = []
    for p in polygons:
        # Skip invalid or empty polygons
        if p is None or p.is_empty or not p.is_valid:
            logger.warning(f'Skipping invalid polygon: {p}')
            continue
        # Skip polygons with too few coordinates
        if len(p.exterior.coords) < 3:
            logger.warning(f'Skipping polygon with insufficient points: {len(p.exterior.coords)}')
            continue
        try:
            grains.append(Grain(np.array(p.exterior.xy), image))
        except Exception as e:
            logger.warning(f'Failed to create grain from polygon: {e}')
            continue
    return grains


def load_grains(fn: str, image: np.ndarray = None) -> list:
    ''' 
    Construct grains from polygrons defined in a GeoJSON file.

    Parameters
    ----------
    fn : str
        Filename for GeoJSON file to read.

    Returns
    -------
    grains : list
        List of Grain objects.
    '''
    grains = polygons_to_grains(segmenteverygrain.read_polygons(fn), image)
    return grains


def save_grains(fn: str, grains: list):
    ''' 
    Save grain boundaries to a GeoJSON file.

    Parameters
    ----------
    fn : str
        Filename for csv to be created.
    grains : list
        List of grains to write to disk.
    '''
    segmenteverygrain.save_polygons([g.polygon for g in grains], fn)


def get_summary(grains: list, px_per_m: float = 1.) -> pd.DataFrame:
    '''
    Summarize grain information as a DataFrame.

    Parameters
    ----------
    grains : list
        List of grains to measure and summarize.
    px_per_m : float, default 1.
        Optional conversion from pixels to meters.

    Returns
    -------
    df : pd.DataFrame
        Dataframe of grain measurements.
    '''
    # Get DataFrame
    df = pd.concat([g.data for g in grains], axis=1).T
    # Convert units
    # HACK: Applies first grain's region_props to all
    for k, d in grains[0].region_props.items():
        if d:
            for col in [c for c in df.columns if k in c]:
                df[col] /= px_per_m ** d
    return df


def save_summary(fn: str, grains: list, px_per_m: float = 1.):
    ''' 
    Save grain measurements as a csv.

    Parameters
    ----------
    fn : str
        Filename for csv to be created.
    grains : list
        List of grains to summarize.
    px_per_m: float, default 1.
        Optional conversion from pixels to meters.
    '''
    summary = get_summary(grains, px_per_m)
    summary.to_csv(fn)
    return summary


def get_histogram(
        grains: list = [],
        px_per_m: float = 1.,
        summary: pd.DataFrame = None) -> tuple[object, object]:
    ''' 
    Produce a histogram of grain size measurements.

    Parameters
    ----------
    grains : list (optional, must provide summary if not given)
        List of grains to measure.
    px_per_m : float, default 1.
        Optional conversion from pixels to meters. Ignored if also passing a
        pre-generated summary.
    summary : pd.DataFrame (optional, must provide grains if not given)
        Grain summary, if already generated from get_summary().

    Returns
    -------
    fig, ax : Matplotlib elements
        Resulting Matplotlib plot.
    '''
    if isinstance(summary, type(None)):
        summary = get_summary(grains, px_per_m)
    # plot_histogram_of_axis_lengths() takes values in mm, not m
    ret = segmenteverygrain.plot_histogram_of_axis_lengths(
        summary['major_axis_length'] * 1000,
        summary['minor_axis_length'] * 1000)
    return ret


def save_histogram(
        fn: str,
        grains: list = [],
        px_per_m: float = 1.,
        summary: pd.DataFrame = None) -> None:
    ''' 
    Save histogram of grain size measurements as an image.

    Parameters
    ----------
    fn : str
        Filename for image to be created. File type will be interpreted.
    grains : list (optional, must provide summary if not given)
        List of grains to represent.
    px_per_m: float, default 1.
        Conversion from pixels to meters. Ignored if also passing a
        pre-generated summary.
    summary : pd.DataFrame (optional, must provide grains if not given)
        Grain summary, if already generated from get_summary().
    '''
    fig, ax = get_histogram(grains, px_per_m=px_per_m, summary=summary)
    fig.savefig(fn, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def get_mask(grains: list, image: np.ndarray) -> np.ndarray:
    ''' 
    Get a rasterized, binary mask of grain shapes as np.ndarray. 

    Parameters
    ----------
    grains : list
        List of grains to represent.
    image : np.ndarray
        Original image.

    Returns
    -------
    np.ndarray
        Binary mask image.
    '''
    polys = [g.polygon for g in grains]
    rasterized_image, mask = segmenteverygrain.create_labeled_image(
        polys, image)
    return keras.utils.img_to_array(mask)


def save_mask(fn: str, grains: list, image: np.ndarray, scale: bool = False):
    '''
    Save binary mask of grain shapes to disk, optionally scaled to 0-255.

    Parameters
    ----------
    fn : str
        Filename for image to be created. File type will be interpreted.
    grains : list
        List of grains to represent.
    image : np.ndarray
        Original image.
    scale : bool
        Whether to scale from 0-255 for human readability (True)
        or 0-1 for model training (False).
    '''
    keras.utils.save_img(fn, get_mask(grains, image), scale=scale)


# Point count ----------------------------------------------------------------

def make_grid(image: np.ndarray, spacing: int) -> tuple[list, list, list]:
    ''' 
    Construct a grid of measurement points given an image and spacing.

    Parameters
    ----------
    image : np.ndarray
        Original image.
    spacing : int
        Spacing between measurement points.

    Returns
    -------
    points : list
        List of shapely.Point objects representing measurement locations.
    xs, ys : lists
        Lists of point coordinates (for convenience).
    '''
    img_y, img_x = image.shape[:2]
    pad_x = img_x % spacing
    pad_y = img_y % spacing
    x_vals = np.arange(round(pad_x / 2), img_x, spacing)
    y_vals = np.arange(round(pad_y / 2), img_y, spacing)
    xs, ys = np.meshgrid(x_vals, y_vals)
    points = shapely.points(xs, ys).flatten()
    return points, xs, ys


def filter_grains_by_points(grains: list, points: list, unique: False) -> tuple[list, list]:
    ''' 
    Generate a list of grains at specified points.

    Parameters
    ----------
    grains : list
        Full list of grains in an image.
    points : list
        List of shapely.Point objects representing measurement locations.
    unique : bool
        Whether the returned list of grains should only contain unique items.
        If True, will remove duplicates. Default False.

    Returns
    -------
    point_grains : list
        Filtered list of grains at specified locations.
    point_found : list
        List representing whether a grain was found at each input point.
    '''
    # Don't modify original list when removing grains with unique == True
    if unique:
        grains = grains.copy()
    # Find grains at given points
    point_grains, point_found = [], []
    for point in points:
        for grain in grains:
            if grain.polygon.contains(point):
                # If results should be unique, remove grain from list
                if unique:
                    grains.remove(grain)
                # Save detected grain
                point_grains.append(grain)
                # Grain found
                point_found.append(True)
                break
        else:
            # Grain not found
            point_found.append(False)
    return point_grains, point_found


def filter_grains_by_props(grains: list, **props):
    for prop, func in props.items():
        filtered_grains = [g for g in filtered_grains if func(g.data[prop])]
    return filtered_grains


# Measurements ---------------------------------------------------------------

def measure_color(image: np.ndarray, polygon: shapely.Polygon) -> dict:
    '''
    Measure color intensities within a polygonal region of an image.

    Parameters
    ----------
    image : np.ndarray
        Input image for analysis.
    polygon : shapely.Polygon
        Polygon defining the region of interest within the image.

    Returns
    -------
    dict
        A dictionary containing the maximum, minimum, and mean intensity values
        for each color channel (R, G, B) within the polygon.
        Keys are formatted as 'max_intensity-0', 'min_intensity-0', 'mean_intensity-0'
        for the red channel, and similarly for green (1) and blue (2) channels.
        This is meant to match the output of skimage.measure.regionprops.
    '''
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    
    # Rasterize polygon into a mask
    bounds = np.array(polygon.bounds)
    xmin, ymin = np.ceil(bounds[:2]).astype(np.int32)
    xmax, ymax = np.floor(bounds[2:]).astype(np.int32)
    
    # Clip bounds to image dimensions
    xmin = max(0, min(xmin, img_width - 1))
    ymin = max(0, min(ymin, img_height - 1))
    xmax = max(0, min(xmax, img_width))
    ymax = max(0, min(ymax, img_height))
    
    # Check if bounds are valid
    if xmax <= xmin or ymax <= ymin:
        logger.warning(f'Polygon bounds are outside image dimensions or invalid: bounds=({xmin},{ymin},{xmax},{ymax}), image=({img_height},{img_width})')
        # Return default values
        return {
            'max_intensity-0': 0,
            'min_intensity-0': 0,
            'mean_intensity-0': 0,
            'max_intensity-1': 0,
            'min_intensity-1': 0,
            'mean_intensity-1': 0,
            'max_intensity-2': 0,
            'min_intensity-2': 0,
            'mean_intensity-2': 0
        }
    
    h, w = ymax - ymin, xmax - xmin
    
    # Clip polygon to image bounds before rasterizing
    image_bounds = shapely.box(0, 0, img_width, img_height)
    clipped_polygon = polygon.intersection(image_bounds)
    
    if clipped_polygon.is_empty:
        logger.warning(f'Polygon is completely outside image bounds')
        return {
            'max_intensity-0': 0,
            'min_intensity-0': 0,
            'mean_intensity-0': 0,
            'max_intensity-1': 0,
            'min_intensity-1': 0,
            'mean_intensity-1': 0,
            'max_intensity-2': 0,
            'min_intensity-2': 0,
            'mean_intensity-2': 0
        }
    
    mask = rasterio.features.rasterize(
        [(clipped_polygon, 1)],
        out_shape=(h, w),
        transform=rasterio.transform.from_bounds(
            xmin, ymax, xmax, ymin, w, h),
        dtype=np.uint8
    )

    # Extract pixel coordinates within the polygon
    y, x = np.where(mask == 1)
    
    # Additional safety check
    if len(y) == 0 or len(x) == 0:
        logger.warning(f'No pixels found within polygon after clipping')
        return {
            'max_intensity-0': 0,
            'min_intensity-0': 0,
            'mean_intensity-0': 0,
            'max_intensity-1': 0,
            'min_intensity-1': 0,
            'mean_intensity-1': 0,
            'max_intensity-2': 0,
            'min_intensity-2': 0,
            'mean_intensity-2': 0
        }
    
    pixels = image[y + ymin, x + xmin]

    # Return information about each color channel
    r, g, b = pixels.T
    return {
        'max_intensity-0': r.max(),
        'min_intensity-0': r.min(),
        'mean_intensity-0': r.mean(),
        'max_intensity-1': g.max(),
        'min_intensity-1': g.min(),
        'mean_intensity-1': g.mean(),
        'max_intensity-2': b.max(),
        'min_intensity-2': b.min(),
        'mean_intensity-2': b.mean()
    }


def measure_polygon(polygon: shapely.Polygon) -> dict:
    '''
    Calculate the area, centroid, and second moments of area of a polygon.

    Parameters
    ----------
    polygon : shapely.Polygon
        The input polygon.

    Returns
    -------
    dict
        A dictionary containing:
        - 'area': Area of the polygon.
        - 'centroid': (Cy, Cx) coordinates of the centroid.
        - 'Ixx': Second moment of area about the centroid (x-axis).
        - 'Iyy': Second moment of area about the centroid (y-axis).
        - 'Ixy': Product of inertia about the centroid.
        This is meant to match the output of skimage.measure.regionprops.
    '''
    # Avoid sign errors by using consistent coordinate order
    polygon = polygon.normalize().reverse()

    # Extract the exterior coordinates of the polygon
    coords = np.array(polygon.exterior.coords)
    x = coords[:-1, 0]
    y = coords[:-1, 1]
    x_next = coords[1:, 0]
    y_next = coords[1:, 1]

    # Calculate the common term for all edges
    common = x * y_next - x_next * y

    # Compute area
    A = 0.5 * np.sum(common)

    # Compute centroid
    Cx = np.sum((x + x_next) * common) / (6 * A)
    Cy = np.sum((y + y_next) * common) / (6 * A)

    # Compute second moments about the origin
    Ixx_origin = np.sum((y**2 + y * y_next + y_next**2) * common) / 12
    Iyy_origin = np.sum((x**2 + x * x_next + x_next**2) * common) / 12
    Ixy_origin = np.sum((x * y_next + 2 * x * y + 2 *
                        x_next * y_next + x_next * y) * common) / 24

    # Transform moments to be about the centroid
    Ixx_centroid = Ixx_origin - A * Cy**2
    Iyy_centroid = Iyy_origin - A * Cx**2
    Ixy_centroid = Ixy_origin - A * Cx * Cy

    return {
        'area': abs(A),
        'centroid': (Cy, Cx),
        'Ixx': abs(Ixx_centroid),
        'Iyy': abs(Iyy_centroid),
        'Ixy': Ixy_centroid
    }


def measure_ellipse(moments: dict) -> dict:
    '''
    Calculate the properties of an ellipse with the given second moments.

    Parameters
    ----------
    moments : dict
        Dictionary containing the second moments of a polygon:
        - 'area': Area of the polygon.
        - 'Ixx': Second moment of area about the x-axis.
        - 'Iyy': Second moment of area about the y-axis.
        - 'Ixy': Product of inertia.

    Returns
    -------
    dict
        A dictionary containing:
        - 'orientation': Orientation of the ellipse in radians (from the x-axis).
        - 'major_axis_length': Full length of the major axis.
        - 'minor_axis_length': Full length of the minor axis.
        This is meant to match the output of skimage.measure.regionprops.
    '''
    Ixx = moments['Ixx']
    Iyy = moments['Iyy']
    Ixy = moments['Ixy']

    # Orientation (theta) in radians
    theta = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)

    # Eigenvalues of the inertia tensor
    common = np.sqrt(((Ixx - Iyy) / 2)**2 + Ixy**2)
    lambda1 = (Ixx + Iyy) / 2 + common  # Major axis eigenvalue
    lambda2 = (Ixx + Iyy) / 2 - common  # Minor axis eigenvalue

    # Major and minor axis lengths
    A = moments['area']
    major_axis_length = 4 * np.sqrt(lambda1 / A)
    minor_axis_length = 4 * np.sqrt(lambda2 / A)

    return {
        'orientation': theta,
        'major_axis_length': major_axis_length,
        'minor_axis_length': minor_axis_length
    }
