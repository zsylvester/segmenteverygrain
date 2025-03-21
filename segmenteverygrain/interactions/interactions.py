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
import segmenteverygrain
import shapely
import skimage
from tqdm import tqdm

# Images larger than this will be downscaled
# 4k resolution is (2160, 4096)
IMAGE_MAX_SIZE = np.asarray((2160, 4096))

# Init logger
logger = logging.getLogger(__name__)

# Speed up rendering a little?
mplstyle.use('fast')

# HACK: Bypass large image restriction
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# HACK: Don't reset zoom level when pressing 'c' to create a grain
if 'c' in mpl.rcParams['keymap.back']:
    mpl.rcParams['keymap.back'].remove('c')

# HACK: Attach parent grain reference to the mpatches.Polygon class
# Makes it easy to use matplotlib "pick" event when clicking on a grain
mpatches.Polygon.grain = None


class Grain(object):
    ''' Stores data and plot representation for a single grain. '''
        
    def __init__(self, xy: np.ndarray, data: pd.Series=None):
        '''
        Parameters
        ----------
        xy : list of (x, y) tuples
            Coordinates to draw this grain as a polygon.
        data : pd.Series (optional)
            Row from a DataFrame containing information about this grain. 
        '''
        
        # Input
        self.data = data
        self.xy = np.array(xy)
        
        # Grain properties to calculate (skimage)
        # {name: dimensionality}
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
            'facecolor': 'lime'
        }
        self.patch = None
        self.selected = False

    @property
    def polygon(self) -> shapely.Polygon:
        ''' 
        Return a shapely.Polygon representing the matplotlib patch.
        
        Returns
        -------
        poly : shapely.Polygon
            Polygon representing the boundaries of this grain.
        '''
        poly = shapely.Polygon(self.xy.T)
        return poly

    def measure(self, image: np.ndarray) -> pd.Series:
        '''
        Calculate grain information from image and matplotlib patch.
        Overwrites self.data.

        Parameters
        ----------
        image : np.ndarray
            Background image, used to calculate region intensity in self.data.

        Returns
        -------
        self.data : pd.Series
            Row for a DataFrame containing computed grain info.
        '''
        # Get rasterized shape
        # TODO: Just use a patch of the image and then convert coords
        rasterized = rasterio.features.rasterize(
            [self.polygon], out_shape=image.shape[:2])
        # Calculate region properties
        data = pd.DataFrame(skimage.measure.regionprops_table(rasterized,
            intensity_image=image, properties=self.region_props.keys()))
        if len(data):
            self.data = data.iloc[0]
        else:
            # TODO: Diagnose why this happens sometimes
            logger.error(f'MEASURE ERROR {pd.DataFrame(data)}')
            self.data = pd.Series()
        return self.data

    def rescale(self, scale: float, save: bool=True) -> pd.Series:
        '''
        Scale polygon coordinates and measurements by given scale factor.

        Parameters
        ----------
        scale : float
            Factor by which to scale grain properties.
        save : bool, default True
            Whether to save the results (True) or just return them (False).
        
        Returns
        -------
        data : pd.Series
            Rescaled grain measurements.
        '''
        # Convert coordinates
        if save:
            self.xy *= scale
        # Convert data
        if type(self.data) is type(None):
            return
        data = self.data if save else self.data.copy()
        # For each type of measured value,
        for k, dim in self.region_props.items():
            # If the value has any length dimensions associated with it
            if dim:
                # Scale those values according to their length dimensions
                for col in [c for c in data.keys() if k in c]:
                    data[col] *= scale ** dim
        return data

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
    def draw_axes(self, ax: mpl.axes.Axes) -> dict[str: object]:
        '''
        Draw centroid and major/minor axes on the provided matplotlib axes.

        Parameters
        ----------
        ax : matplotlib.Axes
            Axes instance on which to draw the grain properties.

        Returns
        -------
        artists : { name : artist }
            Dict of Matplotlib artists, named by property that they represent.
        '''
        # Compute grain data if it hasn't been done already
        if self.data is None:
            image = ax.get_images()[0].get_array()
            self.data = self.measure(image)
        data = self.data
        # Keep track of drawn objects
        artists = {}
        # Centroid
        x0, y0 = data['centroid-1'], data['centroid-0']
        artists['centroid'] = ax.plot(x0, y0, '.k')
        # Major axis
        orientation = data['orientation']
        x = x0 - np.sin(orientation) * 0.5 * data['major_axis_length']
        y = y0 - np.cos(orientation) * 0.5 * data['major_axis_length']
        artists['major'] = ax.plot((x0, x), (y0, y), '-k')
        # Minor axis
        x = x0 + np.cos(orientation) * 0.5 * data['minor_axis_length']
        y = y0 - np.sin(orientation) * 0.5 * data['minor_axis_length']
        artists['minor'] = ax.plot((x0, x), (y0, y), '-k')
        # Return dict of artist objects, potentially useful for blitting
        return artists

    def draw_patch(self, ax: mpl.axes.Axes) -> mpatches.Polygon:
        '''
        Draw this grain on the provided matplotlib axes and save the result.

        Parameters
        ----------
        ax : matplotlib.Axes
            Axes instance on which to draw this grain.

        Returns
        -------
        patch
            Object representing this grain on the plot.
        '''
                
        # Create patch (filled polygon)
        (patch,) = ax.fill(
            *self.xy,
            edgecolor='black',
            linewidth=2.0,
            picker=True,
            animated=True,
            **self.default_props)
        self.patch = patch
        # HACK: Save reference to parent grain within the patch itself
        patch.grain = self
        # Save assigned color (for select/unselect)
        self.default_props['facecolor'] = patch.get_facecolor()
        # Compute grain data for the info box
        if self.data is None:
            image = ax.get_images()[0].get_array()
            self.measure(image)
        return patch


class GrainPlot(object):
    ''' Interactive plot to create, delete, and merge grains. '''

    def __init__(self,
            grains: list = [],
            image: np.ndarray = None, 
            predictor = None,
            blit: bool = True,
            px_per_m: float = 1.,       # px/m
            scale_m: float = 1.,        # m
            minspan: int = 10,          # px
            image_alpha: float = 1.,
            image_max_size: tuple = IMAGE_MAX_SIZE,
            **kwargs):
        '''
        Parameters
        ----------
        grains : list
            List of grains with xy data to plot over the backround image.
        image : np.ndarray
            Image under analysis, displayed behind identified grains.
        predictor
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
        kwargs : dict
            Keyword arguments to pass to plt.figure().
        '''
        logger.info('Creating GrainPlot...')

        # Events
        self.cids = []
        self.events = {
            'button_press_event': self.onclick,
            'button_release_event': self.onclickup,
            'draw_event': self.ondraw,
            'key_press_event': self.onkey,
            'key_release_event': self.onkeyup,
            'pick_event': self.onpick}
        
        # Interaction history
        self.ctrl_down = False
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
                scale = np.max(max_size / image.shape[:2])
                self.display_image = skimage.transform.rescale(
                    image, scale, anti_aliasing=True, channel_axis=2)
                self.scale = scale
                logger.info(f'Downscaled image to {scale} of original.')
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
            onselect = lambda *args: None,  # Don't do anything on selection
            minspanx = minspan,             # Minimum selection size
            minspany = minspan,   
            useblit = True,                 # Always try to use blitting
            props = {
                'facecolor': 'lime',
                'edgecolor': 'black',
                'alpha': 0.2,
                'fill': True},
            spancoords = 'pixels',
            button = [1],                   # Left mouse button only
            interactive = True,
            state_modifier_keys = {})       # Disable shift/ctrl modifiers
        self.box_selector.set_active(False)
        # Replace RectangleSelector update methods to avoid redundant blitting
        if blit:
            self.box_selector.update = self.update
            self.box_selector.update_background = lambda *args: None

        # Scale bar selector
        self.px_per_m = px_per_m
        self.scale_m = scale_m
        self.scale_selector = mwidgets.RectangleSelector(
            self.ax,
            onselect = self.onscale,
            minspanx = minspan,
            minspany = minspan,
            useblit = True,
            props = {
                'facecolor': 'blue',
                'edgecolor': 'black',
                'alpha': 0.2,
                'fill': True},
            spancoords = 'pixels',
            button = [2],
            interactive = True,
            state_modifier_keys = {})
        self.scale_selector.set_active(True)
        # Replace RectangleSelector update methods to avoid redundant blitting
        if blit:
            self.scale_selector.update = self.update
            self.scale_selector.update_background = lambda *args: None

        # Info box
        self.info = self.ax.annotate('',
            xy=(0, 0),
            xytext=(0, 0),
            textcoords='offset points',
            ha='center',
            va='center',
            bbox={'boxstyle': 'round', 'fc':'w'}, 
            animated=blit)
        self.info_grain = None
        self.info_grain_candidate = None

        # Draw grains and initialize plot
        logger.info('Drawing grains.')
        self.grains = grains            # property; sets self._grains
        for grain in tqdm(self._grains):
            grain.draw_patch(self.ax)
        if blit:
            self.artists = [self.info,
                *self.box_selector.artists,
                *self.scale_selector.artists]
            self.canvas.draw()
        logger.info('GrainPlot created!')

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
        artists = ([g.patch for g in self.selected_grains]
            + info_patch
            + self.points
            + self.artists)
        for a in artists:
            self.ax.draw_artist(a)
        # Push to canvas
        self.canvas.blit(self.ax.bbox)

    # Measurements -----------------------------------------------------------
    def onscale(self, eclick: mpl.backend_bases.MouseEvent, erelease: mpl.backend_bases.MouseEvent):
        '''
        Update displayed units based on the selected scale bar length.
        
        Parameters
        ----------
        eclick: MouseEvent
            Details for mouse button down event.
        erelease: MouseEvent
            Details for mouse button release event.
        '''
        # Get length of selection box diagonal in pixels
        x = abs(erelease.xdata - eclick.xdata)
        y = abs(erelease.ydata - eclick.ydata)
        px = np.sqrt(x*x + y*y)
        # Verify that the selection is big enough
        if px < self.minspan:
            self.scale_selector.clear()
            return
        # Convert to pixels per meter using a known scale bar length
        px_per_m = px / self.scale_m
        self.px_per_m = px_per_m
        logger.info(f'Scale set to {px_per_m:f} pixels per meter.')
        # Update the info box using the new units
        self.update_info(self.info_grain)

    def clear_info(self):
        ''' Clear the grain info box. '''
        self.update_info(None)

    def update_info(self, grain=None):
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
        # Set color of new grain
        grain.patch.set_facecolor('blue')
        # Determine box position offset based on grain's position within plot
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
            data = grain.rescale(1 / self.px_per_m, save=False)
            text = (f"Major: {data['major_axis_length']:.3} m\n"
                    f"Minor: {data['minor_axis_length']:.3} m\n"
                    f"Area: {data['area']:.3} m$^2$")
        info.set_text(text)
        # Show info box
        self.info.set_visible(True)

    # Selection helpers ------------------------------------------------------
    def toggle_box(self, active: bool=None) -> bool:
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
    
    def set_point(self, xy: tuple[int, int], foreground: bool=True) -> mpatches.Circle:
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
        new_point = mpatches.Circle(xy,
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
    @property
    def grains(self) -> list:
        ''' 
        Return copy of self.grains in full-image coordinates.
        Necessary when full and display images are different resolutions.
        
        Returns
        -------
        grains : list
            List of saved Grains, converted to full-image coordinates.
        '''
        grains = self._grains.copy()
        if self.scale != 1.:
            for grain in grains:
                grain.rescale(1 / self.scale)
        return grains

    @grains.setter
    def grains(self, grains: list):
        ''' 
        Save copy of given grains list in display-image coordinates.
        
        Parameters
        ----------
        grains : list
            List of grains to copy, rescale, and save.
        '''
        grains = grains.copy()
        if self.scale != 1.:
            for grain in grains:
                grain.rescale(self.scale)
        self._grains = grains

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
            xmin, xmax, ymin, ymax = np.asarray(self.box_selector.extents) / self.scale
            box = np.asarray((xmin, ymin, xmax, ymax))
        else:
            # Return if we haven't provided any prompts
            if points is None:
                return
            box = None
        # Use prompts to find a grain
        coords = segmenteverygrain.predict_from_prompts(
            predictor=self.predictor,
            box=box,
            points=points,
            point_labels=point_labels)
        # Scale and record new grain (on plot, data, and undo list)
        new_grain = Grain(coords)
        new_grain.rescale(self.scale)
        new_grain.draw_patch(self.ax)
        self._grains.append(new_grain)
        self.created_grains.append(new_grain)
        # Clear prompts and update background
        self.clear_all()
        if self.blit:
            self.canvas.draw()
        return new_grain

    def delete_grains(self):
        ''' Delete all selected grains. '''
        # Verify that at least one grain is selected
        if len(self.selected_grains) < 1:
            return
        # Remove selected grains from plot, data, and undo list
        for grain in self.selected_grains:
            grain.patch.remove()
            self._grains.remove(grain)
            if grain in self.created_grains:
                self.created_grains.remove(grain)
        # Clear any prompts (assumed accidental) and update background
        self.clear_all()
        if self.blit:
            self.canvas.draw()

    def hide_grains(self, hide: bool=True):
        ''' 
        Hide or unhide selected grains.
        
        Parameters
        ----------
        hide : bool
            Whether to hide (True) or unhide (False) the selected grains.

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
        # Show/hide selected grains
        for grain in self.selected_grains:
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
        new_grain = Grain(poly.exterior.xy)
        new_grain.draw_patch(self.ax)
        self._grains.append(new_grain)
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
        
        Parameters
        ----------
        event : MouseEvent
            Event details
        '''
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
        # Right click: Background prompt
        elif button == 3:
            self.set_point(coords, False)
        # Only update display if something happened
        else:
            # Forget any grain previously picked with scrollwheel click
            self.info_grain_candidate = None
            return
        self.update()

    def onclickup(self, event: mpl.backend_bases.MouseEvent):
        '''
        Handle click release events anywhere on plot.
        Displays info about any grain indicated with a middle click.
        
        Parameters
        ----------
        event : MouseEvent
            Event details
        '''
        # Only respond to scrollwheel release without dragging
        if (event.button != 2
                or self.last_pick != (round(event.xdata), round(event.ydata))):
            return
        # Update info box and display
        self.update_info(self.info_grain_candidate)
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
        c: Create a grain from existing prompts.
        d: Delete a selected grain.
        m: Create a new grain by merging selected grains.
        z: Undo the most recently created grain.
        control (hold): Temporarily hide selected grains.
        escape: Remove all selections, prompts, and info.
        shift: Activate grain selection box prompt.
        
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
        elif key == 'm':
            self.merge_grains()
        elif key == 'z':
            self.undo_grain()
        elif key == 'control':
            self.ctrl_down = True
            self.hide_grains()
        elif key == 'escape':
            self.clear_all()
        elif key == 'shift':
            self.toggle_box(True)
        # Only update display if something happened
        else:
            return
        self.update()

    def onkeyup(self, event: mpl.backend_bases.KeyEvent):
        ''' 
        Handle key releases.
        Unhides hidden grains or deactivates box selector.
        
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
        elif key == 'shift':
            # Deactivate box selector
            self.toggle_box(False)
            # Cancel box if too small (based on minspan)
            xmin, xmax, ymin, ymax = self.box_selector.extents
            if min(abs(xmax-xmin), abs(ymax-ymin)) < self.minspan:
                self.box_selector.clear()
        # Only update display if something happened
        else:
            return
        self.update()

    def onpick(self, event: mpl.backend_bases.PickEvent):
        '''
        Handle clicking on an existing grain.
        Left-click: Select/unselect the grain.
        Middle-click: Show measurement info for indicated grain.
        
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
        # Save click location for reference by onclick / onclickup
        self.last_pick = (round(mouseevent.xdata), round(mouseevent.ydata))
        # Left-click: Select grain if no point prompts exist
        button = mouseevent.button
        if button == 1 and len(self.points) == 0:
            # Add/remove selected grain to/from selection list
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


# Load/save ---
def load_image(fn: str) -> np.ndarray:
    ''' 
    Load an image from disk as a numpy array.
    
    Parameters
    ----------
    fn : str
        Filename for image to load.

    Returns
    -------
    image : np.ndarray
        Memory representation of loaded image.
    '''
    image = np.array(keras.utils.load_img(fn))
    return image


def polygons_to_grains(polygons: list) -> list:
    ''' 
    Construct grains from a list of polygons defining grain boundaries.
    
    Parameters
    ----------
    polygons : list of shapely.Polygon
        Polygons defining grain boundaries.

    Returns
    -------
    grains : list
        List of Grain objects.
    '''
    grains = [Grain(np.array(p.exterior.xy)) for p in polygons]
    return grains


def load_grains(fn: str) -> list:
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
    grains = polygons_to_grains(segmenteverygrain.read_polygons(fn))
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


def get_summary(grains: list, px_per_m: float=1.) -> pd.DataFrame:
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
    df = df.reset_index(drop=True)
    # Convert units
    # HACK: Applies first grain's region_props to all
    for k, d in grains[0].region_props.items():
        if d:
            for col in [c for c in df.columns if k in c]:
                df[col] /= px_per_m ** d 
    return df


def save_summary(fn: str, grains: list, px_per_m: float=1.):
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
    get_summary(grains, px_per_m).to_csv(fn)


def get_histogram(grains: list, px_per_m: float=1.) -> tuple[object, object]:
    ''' 
    Produce a histogram of grain size measurements.
    
    Parameters
    ----------
    grains : list
        List of grains to measure.
    px_per_m : float, default 1.
        Optional conversion from pixels to meters.

    Returns
    -------
    fig, ax : Matplotlib elements
        Resulting Matplotlib plot.
    '''
    df = get_summary(grains, px_per_m)
    fig, ax = segmenteverygrain.plot_histogram_of_axis_lengths(
        df['major_axis_length'], df['minor_axis_length'])
    return fig, ax
    

def save_histogram(fn: str, grains: list, px_per_m: float=1.):
    ''' 
    Save histogram of grain size measurements as an image.
    
    Parameters
    ----------
    fn : str
        Filename for image to be created. File type will be interpreted.
    grains : list
        List of grains to represent.
    px_per_m: float, default 1.
        Optional conversion from pixels to meters.
    '''
    fig, ax = get_histogram(grains, px_per_m)
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
    mask : np.ndarray
        Binary mask image.
    '''
    polys = [g.polygon for g in grains]
    rasterized_image, mask = segmenteverygrain.create_labeled_image(
        polys, image)
    mask = keras.utils.img_to_array(mask)
    return mask


def save_mask(fn: str, grains: list, image: np.ndarray, scale: bool=False):
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


# Point count ---
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


def filter_grains_by_points(grains: list, points: list) -> tuple[list, list]:
    ''' 
    Generate a list of grains at specified points.
 
    Parameters
    ----------
    grains : list
        Full list of grains in an image.
    points : list
        List of shapely.Point objects representing measurement locations.

    Returns
    -------
    point_grains : list
        Filtered list of grains at specified locations.
    point_found : list
        List representing whether a grain was found at each input point.
    ''' 
    point_found = []
    point_grains = []
    for point in points:
        for grain in grains.copy():
            if grain.polygon.contains(point):
                # Remove grain from list so that it's not found twice
                grains.remove(grain)
                # Save detected grain
                point_grains.append(grain)
                # Record that a grain was found at this point
                point_found.append(True)
                break
        else:
            # Record that no grain was found at this point
            point_found.append(False)
    return point_grains, point_found


def filter_grains_by_props(grains: list, **props):
    for prop, func in props.items():
        filtered_grains = [g for g in filtered_grains if func(g.data[prop])]
    return filtered_grains