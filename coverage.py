import pandas as pd
import numpy as np
import pyranges as pr
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import os, itertools
from typing import Union


class ObjectInitialization:
    def __init__(self, **kwargs) -> None:
        # initialize arguments
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # check if essential arguments have been provided
        self._validate_minimal_requirements()
        self._define_if_peaks_is_selected()
        self._process_strand_colors()
        
        # inputs formatting
        self._process_bgzf_files()
        self._process_bgzf_files_colors()
        
        # initialize and process object required by the other methods
        self._to_low_cap_list_annotation_and_peaks_names()
        self.plot_types_proportions_adjusted = self.plot_types_proportions.copy()
        self._process_region()
        self.selected_plots = []
        self.axis_count = 0


    def _validate_minimal_requirements(self) -> None:
        supplied_count = (
            (self.links is not None) +
            (self.annotation is not None) +
            isinstance(self.peaks, pd.DataFrame) +
            isinstance(self.peaks, pr.PyRanges) +
            (isinstance(self.peaks, bool) and self.peaks) +
            (self.bgzf_files is not None)
        )
        assert supplied_count > 0, (
            '_validate_minimal_requirements: ' + 
            'Must specify at least 1 of the arguments: '+
            'links, annotation, peaks or bgzf_files'
        )
        
    
    def _define_if_peaks_is_selected(self) -> None:
        is_data_frame = isinstance(self.peaks, pd.DataFrame)
        is_ranges = isinstance(self.peaks, pr.PyRanges)
        is_bool = isinstance(self.peaks, bool)
        if is_data_frame or is_ranges or (is_bool and self.peaks):
            self.peaks_selected = True
        else:
            self.peaks_selected = False
        
    
    def _to_low_cap_list(self, attr: str) -> None:
        obj = getattr(self, attr)
        if isinstance(obj, tuple):
            obj = list(obj)
        if isinstance(obj, list):
            obj = list(map(lambda x: x.lower(), obj))
            setattr(self, attr, obj)
        elif isinstance(obj, str):
            obj = [obj.lower()]
            setattr(self, attr, obj)
            
    def _to_low_cap_list_annotation_and_peaks_names(self) -> None:
        attrs = ['annotation_names', 'peaks_names']
        list(map(self._to_low_cap_list, attrs))
    
    
    def _process_extend_region(self) -> None:
        if isinstance(self.extend_region, int):
            self.extend_region = [self.extend_region, self.extend_region]
        if isinstance(self.extend_region, tuple):
            self.extend_region = list(self.extend_region)
        if isinstance(self.extend_region, list):
            self.extend_region = dict(zip(['start', 'end'], self.extend_region))
        if not isinstance(self.extend_region, dict):
            raise ValueError(
                '_process_extend_region: extend_region must be list, tuple or int. ' +
                f'Got: {type(self.extend_region)}'
            )
            
    def _extend_region(self) -> None:
        self._process_extend_region()
        start_diff = self.region['start'] - self.extend_region['start']
        self.region['start'] = max(0, start_diff)
        self.region['end'] = self.region['end'] + self.extend_region['end']
                 
    def _split_region(self) -> None:
        import re
        self.region = self.region.replace(',', '') # strip possible commas
        self.region = re.split(':|-|_', self.region)
    
    def _process_region(self) -> None:
        if isinstance(self.region, tuple):
            self.region = list(self.region)
        if isinstance(self.region, str):
            self._split_region()
        if len(self.region) != 3:
            raise ValueError(
                f'_process_region: coordinates ' +
                f'must contain 3 elements. Got {self.region}'
            )
        if isinstance(self.region, list):
            keys = ['chr', 'start', 'end']
            self.region = dict(zip(keys, self.region))
        
        self.region['start'] = int(self.region['start'])
        self.region['end'] = int(self.region['end'])
        
        if self.region['end'] < self.region['start']:
            raise ValueError(
                '_process_region: end < start. ' +
                f"Got: {self.region['end']} < {self.region['start']}"
            ) 
            
        if self.extend_region:
            self._extend_region()
        
        self.chr = self.region['chr']
        self.start = self.region['start']
        self.end = self.region['end']
        self.region_width = self.region['end'] - self.region['start']
        
        
            
    def _process_bgzf_files(self) -> None:
        if self.bgzf_files is None:
            return None
        if isinstance(self.bgzf_files, dict):
            return None   
        if isinstance(self.bgzf_files, str):
            self.bgzf_files = [self.bgzf_files]
        if isinstance(self.bgzf_files, tuple):
            self.bgzf_files = list(self.bgzf_files)
        if isinstance(self.bgzf_files, list):
            names = map(lambda x: os.path.basename(x).split('.')[0], self.bgzf_files)
            self.bgzf_files = dict(zip(names, self.bgzf_files))
            
            
    def _process_bgzf_files_colors(self) -> None:
        if self.bgzf_files is None:
            return None
        if isinstance(self.bgzf_files_colors, dict):
            return None   
        if isinstance(self.bgzf_files_colors, str):
            self.bgzf_files_colors = [self.bgzf_files_colors] * len(self.bgzf_files)
        if isinstance(self.bgzf_files_colors, tuple):
            self.bgzf_files_colors = list(self.bgzf_files_colors) 
        if isinstance(self.bgzf_files_colors, list):
            assert len(self.bgzf_files_colors) == len(self.bgzf_files), (
                '_process_bgzf_files_colors: bgzf_files and ' +
                'bgzf_files_colors must have the same length'
            )
            self.bgzf_files_colors = dict(zip(self.bgzf_files.keys(), self.bgzf_files_colors))

            
    def _process_strand_colors(self) -> None:
        if self.strand_colors is None:
            return None
        if isinstance(self.strand_colors, dict):
            return None 
        if isinstance(self.strand_colors, str):
            self.strand_colors = [self.strand_colors] * 2
        if isinstance(self.strand_colors, tuple):
            self.strand_colors = list(self.strand_colors) 
        if isinstance(self.strand_colors, list):
            assert len(self.strand_colors) == 2, (
                '_process_strand_colors ' +
                'strand_colors must have length == 2'
            )
            self.strand_colors = dict(zip(['+', '-'], self.strand_colors))

                
    def __repr__(self):
        c = '\n\nObject of class:\t' + type(self).__name__ + '\n\n'
        o = self.__dict__
        out = ''        
        for k_, o_ in o.items():
            out += k_ + ':\t' + str(type(o_)) + '\n'  
        o_additional = {}    
        for o_ in dir(self):
            if '__' not in o_ and (o_ not in o.keys()):
                obj = getattr(self, o_)
                o_additional[o_] = str(type(obj))
        out_additional = ''        
        for k_, o_ in o_additional.items():
            out_additional += k_ + ':\t' + repr(o_) + '\n'
        return c + out + out_additional


class ComponentsProcessing:
    def generate_objects(self) -> None:
        # Generate objects
        # Links
        # It must be first in the hyerarchy because
        # the region must be extended before the 
        # annotation and the peaks in case the 
        # relevant option is provided by the user.
        self.generate_links_object()
        # Annotation
        self.generate_annotation_object()
        # Peaks
        self.generate_peaks_object()
        # Coverage
        self.generate_coverage_objects()
        # This method is very specific and therefore
        # it is provided on the main class

        # Compute figure settings
        self._reorder_plot_types()
        self._compute_axis_heights()


class Links:
    # Links methods
    def _validate_links_overflowing_method(self) -> None:
        if self.links_overflowing_method is None:
            return None
        options = ["extend_region", "remove_overflowing"]
        if self.links_overflowing_method in options:
            return None
        raise ValueError(
            f'_validate_links_overflowing_method: ' +
            f'links_overflowing_method must be: {options + [None]}'+
            f'Provided: {self.links_overflowing_method}'
        )
            
    def _subset_links(self) -> None:
        '''
        Needs to be a dedicated method because _subset_ranges
        can perform 2 rounds of overlap with the region and 
        it can clip the ranges (which is not wanted for links).
        '''
        links = self.links
        if isinstance(links, pd.DataFrame):
            links = pr.PyRanges(links, int64 = True)
        region_pyranges = self._generate_region_pyranges()
        links = links.overlap(region_pyranges)
        if len(links) == 0:
            self.links = None
            return None
        links = links.as_df()
        self._validate_links_overflowing_method()
        if self.links_overflowing_method == 'extend_region':
            self.start = np.min(links.loc[:, ['Start', 'End']].values)
            self.end = np.max(links.loc[:, ['Start', 'End']].values)
        if self.links_overflowing_method == 'remove_overflowing':
            links = links.query('Start >= @self.start & End <= @self.end')
        if len(links) == 0:
            self.links = None
            return None
        self.links = links
    
    
    def _validate_links_orientation(self) -> None:
        options = ['top', 'bottom']
        if self.links_orientation in options:
            return None
        raise ValueError(
            f'_validate_links_orientation: links_orientation must be one of: ' +
            f'{options}. Provided: "{self.links_orientation}"'
        )
    
    def _validate_scale_attribute_type(self, 
                                       attribute_name: str, 
                                       attribute: Union[int, float, str]) -> None:
        types = [int, float, str]
        if type(attribute) in types:
            return None
        raise ValueError(
            f'_validate_scale_attribute_type: {attribute_name} '+
            f'must be: {types}. Provided {type(attribute)}'
        )
        
    def _validate_scale_attribute_float(self, 
                                        attribute_name: str, 
                                        attribute: float) -> None:
        if (attribute >= 0) and (attribute <= 1):
            return None
        raise ValueError(
            f'_validate_scale_attribute_float: {attribute_name} must be ' +
            f'between 0 and 1. Provided: {attribute}'
        )
    
    def _validate_scale_attribute_str(self, 
                                      attribute_name: str, 
                                      attribute: str) -> None:
        if attribute in self.links.columns:
            return None
        raise ValueError(
            f'_validate_scale_attribute_str: {attribute_name} must be '+
            f'a column of the links object: {self.links.columns}. '+
            f'Provided {attribute}'
        )
        
    def _append_scale_column(self, 
                             attribute_name: str, 
                             links: pd.DataFrame, 
                             column_name: str) -> None:
        '''
        The scale column is added in place on the provided links object.
        '''
        attribute = getattr(self, attribute_name)
        if attribute is None:
            links.loc[:, column_name] = 1 
            return None
        self._validate_scale_attribute_type(attribute_name, attribute)
        if isinstance(attribute, int):
            attribute = float(attribute)
        if isinstance(attribute, float):
            self._validate_scale_attribute_float(attribute_name, attribute)
            links.loc[:, column_name] = attribute
        if isinstance(attribute, str):
            self._validate_scale_attribute_str(attribute_name, attribute)
            links_scale_ptp = self.links_scale[1] - self.links_scale[0]
            links.loc[:, column_name] = links.loc[:, attribute] - self.links_scale[0] 
            links.loc[:, column_name] /= links_scale_ptp
    
    def _validate_links_min_height(self) -> None:
        if  (self.links_min_height >= 0) and (self.links_min_height <=1):
            return None
        raise ValueError(
            f'_validate_links_min_height: links_min_height must be between 0 and 1' +
            f'Provided: {self.links_min_height}'
        )

    def _scale_ellipse_height(self, links: pd.DataFrame) -> None:
        if not self.links_proportional_height:
            return None
        if len(links) <= 1:
            return None
        self._validate_links_min_height()
        width = links.loc[:, 'Width'].values # ndarray
        width_ptp = np.ptp(width) # scalar
        if width_ptp <= 1:
            return None
        width_min = np.min(width) # scalar
        y_axis_prop_ptp = 1 - self.links_min_height # scalar
        width_diff = width - width_min # ndarray
        height = y_axis_prop_ptp * width_diff / width_ptp
        height += self.links_min_height
        links.loc[:, 'Ellipse_height'] = height
    
    def _generate_ellipse(self, index, row: pd.Series) -> matplotlib.patches.Ellipse:
        width = row.Width
        xy = (row.Center, 0)
        alpha = row.Alpha
        linewidth = row.Linewidth
        ellipse_height = row.Ellipse_height
        edgecolor = matplotlib.colors.colorConverter.to_rgba(
            self.links_color, alpha = alpha
        )
        ellipse = matplotlib.patches.Ellipse(
            xy=xy, width=width, height=ellipse_height, 
            facecolor='none', edgecolor=edgecolor, 
            linewidth=linewidth
        )
        return ellipse
        
    def plot_links(self, ax: 'matplotlib.axes._subplots.AxesSubplot' = None) -> None:
        # Inputs processing
        if self.links is None:
            return None  
        self._validate_links_orientation()
        # links processing
        links = self.links.copy()
        self._append_scale_column(
            attribute_name='links_alpha', 
            links=links,  column_name='Alpha'
        ) 
        self._append_scale_column(
            attribute_name='links_linewidth', 
            links=links,  column_name='Linewidth'
        ) 
        links.loc[:, 'Linewidth'] *= self.links_linewidth_multiplier    
        links.loc[:, 'Center'] = np.mean(links.loc[:, ['Start', 'End']], axis = 1)
        links.loc[:, 'Width'] = np.abs(links.loc[:, 'End'] - links.loc[:, 'Start'])
        links.loc[:, 'Ellipse_height'] = 1
        self._scale_ellipse_height(links)
        # Axis settings
        ax.set_ylim(0, 0.51) # 0.51 is to allow 
        # for some space in case the linewidth is increased
        ax.axis('off')
        if self.links_orientation == 'bottom':
            ax.invert_yaxis()
        # Plot (add ellipses to preexisting axes)
        ellipses = map(self._generate_ellipse, *zip(*links.iterrows()))
        list(map(lambda ellipse: ax.add_patch(ellipse), ellipses))

    def generate_links_object(self):
        # Links
        if self.links is  None:
            return None
        self._subset_links()
        if (self.links is None) and not self.keep_empty_axis:
            return None
        # None is returned by _subset_links in case the DataFrame has no rows
        # It does not have the same meaning as the testing done above (where
        # it is checked whether the annotation argument has been provided by the user)
        self.selected_plots += ['links']
        self.axis_count += 1


class Features:
    def _generate_peaks_data_frame(self) -> None: 
        '''
        Modifies the object in place by adding 
        a peaks object (it overrides the 
        user-supplied peaks argument)
        '''
        if not (isinstance(self.peaks, bool) and self.peaks):
            return None
        peaks = (
            self.adata
            .var
            .index
            .to_series()
            .str
            .split('_', expand = True)
            .set_axis(['Chromosome', 'Start', 'End'], axis = 1)
            .rename_axis('peak_name', axis = 0)
            .reset_index()
        )
        self.peaks = pr.PyRanges(peaks, int64 = True)

    def _validate_provided_peaks_object(self) -> None:
        '''
        Stop execution with an assertion error in case
        the required columns are not provided.
        Does nothing otherwise.
        '''
        is_data_frame = isinstance(self.peaks, pd.DataFrame)
        is_pyranges = isinstance(self.peaks, pr.PyRanges)
        if not (is_data_frame or is_pyranges):
            return None
        required_cols = ['Chromosome', 'Start', 'End']
        if self.peaks_names:
            required_cols += ['peak_name']
        peaks_cols = self.peaks.columns # needs to be a pandas object
                                        # for the vectorized check below 
        has_cols = all([_ in peaks_cols for _ in required_cols])
        assert(has_cols), (
            '_validate_provided_peaks_object: ' +
            'Provided peaks object must contain the columns:'+                 
            f' {required_cols}. Got instead: {peaks_cols.to_list()}'
        )

    # Ranges processing methods
    def _find_clipped_genes_at_start(self, ranges: pd.DataFrame) -> list:
        ranges = ranges.copy()
        if isinstance(ranges, pr.PyRanges):
            ranges = ranges.as_df()
        starts = ranges.Start.values
        ends = ranges.End.values
        strands = ranges.Strand.values
        affected = (
            ((starts < self.start) & (strands == '+')) | 
            ((ends > self.end) & (strands == '-'))
        )
        clipped_genes_at_start = (
            ranges
            .loc[affected, 'gene_name']
            .unique()
            .tolist()
        )
        return clipped_genes_at_start
        
        
    def _clip_ranges(self, ranges: pd.DataFrame) -> pd.DataFrame:
        '''
        The chromosome will not be checked because 
        the ranges are assumed to be already filtered.
        '''
        ranges = ranges.copy()

        starts = ranges.Start.values
        ends = ranges.End.values

        ranges['Start'] = np.select(
            [starts < self.start, starts > self.start], 
            [np.repeat(self.start, len(starts)), starts]
        )
        ranges['End'] = np.select(
            [ends > self.end, ends < self.end], 
            [np.repeat(self.end, len(ends)), ends]
        )

        return ranges

    
    def _generate_region_pyranges(self) -> pr.PyRanges:
        ranges = pd.DataFrame({
            "Chromosome": [self.chr], 
            "Start": [self.start], 
            "End": [self.end]
        })
        ranges = pr.PyRanges(ranges, int64 = True)
        return ranges
    
    
    def _subset_ranges(self, 
                       ranges: pd.DataFrame, 
                       clip: bool, 
                       store_clipped: bool) -> pd.DataFrame:
        '''
        The subsetting can take place in 1 or 2 steps depending
        on the "clip" argument.
        If clip == True, then the ranges are subsetted and the 
        rangse that overlap with the extremities of the selected
        region are trimmed.
        If clip == False, then a first subsetting is performed,
        the extremities (if any) of the ranges will override
        the start and end of the selected region. After this,
        a subsequent subsetting is performed with the new bounds.
        This additional subsetting serves the function of rescuing
        possible internal ranges (for example exons) which are
        filtered out during the first subsetting (because they 
        are completely contained within a larger range, normally
        a transcript or a gene range).
        '''
        ranges = ranges.copy()
        if isinstance(ranges, pd.DataFrame):
            ranges = pr.PyRanges(ranges, int64 = True)

        region_pyranges = self._generate_region_pyranges()
        ranges_intermediate = ranges.overlap(region_pyranges)
        
        if len(ranges_intermediate) == 0:
            return None
        if clip and store_clipped:
            self.clipped_genes_at_start = self._find_clipped_genes_at_start(
                ranges_intermediate
            )
        if clip and not store_clipped:
            ranges_output = self._clip_ranges(ranges_intermediate.as_df())
        if not clip:
            self.start = min(np.min(ranges_intermediate.Start), self.start)
            self.end =  max(np.max(ranges_intermediate.End), self.end)
            self.region_width = self.end - self.start
            region_pyranges = self._generate_region_pyranges()
            ranges_output = ranges.overlap(region_pyranges)
            ranges_output = ranges_output.as_df()
        return ranges_output


    def generate_peaks_object(self) -> None:
        if not self.peaks_selected:
            return None
        self._generate_peaks_data_frame()
        self._validate_provided_peaks_object()
        self.peaks = self._subset_ranges(
            ranges = self.peaks, 
            clip = self.peaks_clip, 
            store_clipped = False
        )
        if (self.peaks is None) and not self.keep_empty_axis:
            return None
        # None is returned by _subset_ranges
        # in case the DataFrame has no rows
        self.selected_plots += ['peaks']
        self.axis_count += 1
                
    def generate_annotation_object(self) -> None:
        if self.annotation is None:
            return None
        self.annotation = self._subset_ranges(
            ranges = self.annotation, 
            clip = self.annotation_clip, 
            store_clipped = True
        )
        if (self.annotation is None) and not self.keep_empty_axis:
            return None
        # None is returned by _subset_ranges in case the DataFrame has no rows
        # It does not have the same meaning as the testing done above (where
        # it is checked whether the annotation argument has been provided by the user)
        self.annotation_arrow_length *= self.region_width
        self.selected_plots += ['annotation']
        self.axis_count += 1
                
            
    @staticmethod
    def plot_boxes(ranges: pd.DataFrame, 
                   ax: 'matplotlib.axes._subplots.AxesSubplot',
                   line_color = 'grey', 
                   line_width: float = 0.5, 
                   height: float = 1, 
                   fill_color = 'lightgrey',
                   y_offset: float = 0) -> None:
        '''
        height: float, fraction of the y axis
        '''
        boxes = (
            ranges
            .loc[:, ['Start', 'End']]
            .assign(width = lambda x: x.End - x.Start)
            .values
            .tolist()    
        )
        def generate_rectangles(start, end, width) -> 'matplotlib.patches.Rectangle':
            rectangle = matplotlib.patches.Rectangle(
                xy = (start, (- height / 2) + y_offset), 
                width = width, 
                height = height, 
                linewidth = line_width, 
                edgecolor = line_color, 
                facecolor = fill_color,
                zorder=10
            )
            return rectangle
        rectangles = list(map(generate_rectangles, *zip(*boxes)))
        list(map(lambda x: ax.add_patch(x), rectangles))


    def plot_annotation(self, ax: 'matplotlib.axes._subplots.AxesSubplot' = None) -> None:
        '''
        Supported functionality only exons by gene
        exon_height: float, fraction of the y axis
        arrow_length: float, fraction of the x axis
        '''
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        # dealing with empty annotation
        if self.annotation is None:
            return None
        # plot group by group
        list(map(
            self._plot_gene_annotation,
            *zip(*self.annotation.groupby(self.annotation_group_by)),
            itertools.repeat(ax)
        ))

    def _plot_gene_annotation(self, 
                              name: str, 
                              group: pd.DataFrame, 
                              ax: 'matplotlib.axes._subplots.AxesSubplot') -> None:
        gene = group.query('Feature == @self.annotation_main_feature')
        gene_start, gene_end, gene_strand = gene[['Start', 'End', 'Strand']].iloc[0]
        gene_range = [gene_start, gene_end]
        if gene_strand == '+':
            y_offset = self.annotation_height / 2
            gene_start = gene_start
            arrow_displacement = self.annotation_arrow_length
            y_text = 3 * y_offset + self.annotation_names_y_spacing
        if gene_strand == '-':
            y_offset = -self.annotation_height / 2
            gene_start = gene_end
            arrow_displacement = - self.annotation_arrow_length
            y_text = 3 * y_offset - self.annotation_names_y_spacing

        ax.plot( # line marking the gene body
            gene_range,
            [y_offset, y_offset],
            color=self.annotation_line_color,
            linewidth=self.annotation_line_width
        )

        exons = group.query('Feature == "exon"')
        if len(exons) > 1 and gene_strand == '+':
            last_exon = exons.iloc[[-1], :]
            exons = exons = exons.iloc[:-1, :]
        if len(exons) > 1 and gene_strand == '-':
            last_exon = exons.iloc[[1], :]
            exons = exons = exons.iloc[1:, :]
        self.plot_boxes( # add exons
            ranges = exons, 
            ax = ax,    
            line_color = self.annotation_line_color, 
            line_width = self.annotation_line_width, 
            height = self.annotation_height, 
            fill_color = self.annotation_fill_color,
            y_offset = y_offset
        )
        if len(exons) > 1:
            self.plot_boxes( # add exons
                ranges = last_exon, 
                ax = ax,    
                line_color = self.annotation_line_color, 
                line_width = self.annotation_line_width, 
                height = self.annotation_height/2, 
                fill_color = self.annotation_fill_color,
                y_offset = y_offset
            )
        
        has_clipped = hasattr(self, 'clipped_genes_at_start')
        is_clipped = has_clipped and name in self.clipped_genes_at_start
        if is_clipped or not self.annotation_arrow:
            return None
        ax.plot( # vertical line above gene start
            [gene_start, gene_start],
            [y_offset, 3 * y_offset],
            color=self.annotation_line_color,
            linewidth=self.annotation_line_width
        )
        ax.arrow( # horizontal arrow
            x=gene_start,
            y=3 * y_offset,
            dx=arrow_displacement,
            dy=0,
            head_width=self.annotation_height / 4,
            head_length=self.annotation_arrow_length,
            facecolor=self.annotation_line_color,
            edgecolor=self.annotation_line_color,
            linewidth=self.annotation_line_width
        )

        is_list = isinstance(self.annotation_names, list)
        has_name = is_list and name.lower() in self.annotation_names
        is_bool = isinstance(self.annotation_names, bool)
        is_all_names = is_bool and self.annotation_names
        if not has_name and not is_all_names:
            return None
        ax.annotate( # gene name
            name.title(),
            xy=(gene_start, y_text),
            xycoords='data',
            xytext=(gene_start, y_text),
            textcoords='data',
            horizontalalignment='center',
            verticalalignment=(gene_strand == '+' and 'bottom' or 'top')
            # color = self.annotation_line_color # Text kwargs
        )
            
            
    def plot_peaks(self, ax: 'matplotlib.axes._subplots.AxesSubplot' = None) -> None:
        ax.set_ylim(-0.51, 0.5) # -0.51 is to show also the bottom line
        ax.axis('off')
        # dealing with empty peaks
        if self.peaks is None:
            return None
        y_offset = - (0.5 - (self.peaks_height / 2))
        y_text = -0.5 + self.peaks_names_y_spacing + self.peaks_height
        peaks_mean = np.mean(self.peaks[['Start', 'End']].values, axis = 1)

        self.plot_boxes( # add peaks
            ranges = self.peaks, 
            ax = ax,    
            line_color = self.peaks_line_color, 
            line_width = self.peaks_line_width, 
            height = self.peaks_height, 
            fill_color = self.peaks_fill_color,
            y_offset = y_offset
        )
        if not self.peaks_names:
             return None
            
        for i in range(self.peaks.shape[0]):
            name = self.peaks['peak_name'][i]
            is_list = isinstance(self.peaks_names, list)
            has_name = is_list and name.lower() in self.peaks_names
            is_bool = isinstance(self.peaks_names, bool)
            is_all_names = is_bool and self.peaks_names
            if not has_name and not is_all_names: 
                return None
            ax.annotate( # add text
                name,
                xy = (peaks_mean[i], y_text), 
                xycoords = 'data',
                xytext = (peaks_mean[i], y_text), 
                textcoords = 'data',
                horizontalalignment = 'center', 
                verticalalignment = 'bottom'
                #color = self.annotation_line_color # Text kwargs
            )

        


class Coverage:
    def _validate_bin_size(self) -> None:
        if self.region_width > self.bin_size:
            return None
        raise ValueError(
            f'_validate_bin_size: '+
            f'range_ size ({self.region_width}) smaller' +
            f' than bin_size ({self.bin_size})' +
            f' increase range_ size or reduce bin_size'
        )

    def create_ranges_bins(self) -> None:
        '''
        Modifies the object in place (adding the bins attribute).

        The bins start from the very 1st nucleotide of the range.
        The end of a bin matches the start of the following bin,
        the reason for this is that pyranges coverage by default 
        treates the start of the range as open and the end as closed.
        The last bin is trimmed at the end to fit the end of the 
        range which may not correspond to the end of the bin.
        '''
        self.n_bins, remainder = np.divmod(
            self.region_width, 
            self.bin_size
        )
        # create bins starting from 0
        if remainder == 0:
            self.bins = np.c_[
                np.arange(0, self.n_bins) * self.bin_size, 
                np.arange(1, self.n_bins + 1) * self.bin_size
            ]
        else:
            self.bins = np.c_[
                np.arange(0, self.n_bins + 1) * self.bin_size, 
                np.arange(1, self.n_bins + 2) * self.bin_size
            ]
            self.bins[-1, 1] = self.bins[-1, 0] + remainder
        # add the start offset of the bins
        self.bins = self.bins + self.start
        self.bins = pd.DataFrame(self.bins, columns = ['Start', 'End'])
        # the middle position of the bin is truncated of the decimal digits
        self.bins['Mid_bin'] = (
            np.mean(self.bins[['Start', 'End']], axis = 1)
            .astype(np.int64)
        )
        self.bins['Chromosome'] = self.chr
 

    def _read_bgzf_file(self, file: str) -> pd.DataFrame:
        import pysam
        columns = [
            'Chromosome', 'Start', 'End', 
            'Barcode', 'Value', 'Strand'
        ]
        tbx = pysam.TabixFile(file)
        reads = [row.split()[:len(columns)] for row in tbx.fetch(self.chr, self.start, self.end)]
        n_reads = len(reads)
        if n_reads > 0:
            n_cols = len(reads[0])
            if n_cols <= len(columns):
                columns = columns[:n_cols]
        bam = (
            pd.DataFrame(reads, columns = columns)
            .drop(columns = 'Value')
        )
        # Adjust the types because they were imported as string by tbx.fetch
        bam['Start'] = bam['Start'].astype(np.int64)
        bam['End'] = bam['End'].astype(np.int64)
        return bam

    
    def read_bgzf_files(self) -> None:
        '''
        Modifies the object inplace by adding a bam attribute.
        Required attributes on self: 
            bgzf_files: str, 
            chr: str,
            start: int, 
            end: int.
        '''
        bams = map(self._read_bgzf_file, self.bgzf_files.values())
        names = self.bgzf_files.keys()
        self.bam = dict(zip(names, bams))
            
    def set_bam_attrubutes(self, bam: pd.DataFrame, name: str) -> pd.DataFrame:
        bam.attrs['Strand'] = '*'
        bam.attrs['ylabel'] = name if self.show_ylabel else None
        bam.attrs['color'] = self.bgzf_files_colors[name]
    
    def set_bam_attrubutes_groued(self) -> None:
        for name, bam in self.bam.items():
            self.set_bam_attrubutes(bam, name)
      
    
    def compute_coverage(self, bam: pd.DataFrame) -> pd.DataFrame:
        '''    
        Coverage = binned count of reads.
        '''
        attrs = bam.attrs.copy()
        # convert the attributes of interest to Pyranges
        bins = pr.PyRanges(self.bins, int64 = True)
        bam = pr.PyRanges(bam, int64 = True)
        coverage = bins.coverage(bam, overlap_col = "Count").as_df()
        coverage.drop(columns = ['Start', 'End', 'FractionOverlaps'], inplace=True)
        coverage.attrs = attrs
        return coverage

    def compute_coverage_grouped(self) -> None:
        coverages = map(self.compute_coverage, self.bam.values())
        names = self.bam.keys()
        self.coverage = dict(zip(names, coverages))
    
    
    def add_size_factors_attribute(self) -> None:
        if self.size_factors is None:
            return None
        if not isinstance(self.size_factors, dict):
            raise TypeError('size_factors must be dict')
        has_keys = all([k in self.size_factors.keys() for k in self.bam.keys()])
        if not has_keys:
            raise KeyError(
                f'add_size_factors_attribute: size_factors keys must be in ' +
                f'{self.bam.keys()}. Provided: {self.size_factors.keys()}'
            )
        for name, bam in self.bam.items():
            bam.attrs['size_factor'] = self.size_factors[name]
            
            
            
    def resize_coverage_grouped(self) -> None:
        '''
        Normalization based on provided size factors
        Take the precedence over "normalize" in case 
        both are specified
        '''
        if not isinstance(self.size_factors, dict):
            return None
        for coverage in self.coverage.values():
            size_factor = coverage.attrs['size_factor']
            coverage.loc[:, 'Count'] = (
                coverage.loc[:, 'Count'] / size_factor
            )
                
    def _validate_normalize_method(self) -> None:
        options = [None, 'rpkm', 'total']
        if self.normalize in options:
            return None
        raise ValueError(
            f"_normalize_coverage: normalize must be " +
            f"one of: {options}. " +
            f"Cannot handle normalize = '{self.normalize}'"
        )
        
    def _verify_total_count(self) -> None:
        if self.total_count:
            return None
        raise ValueError(
            '_normalize_coverage: total_count must be ' +
            'provided when using normalize = "total"'
        )
        
    def _normalize_coverage(self, coverage: pd.DataFrame) -> pd.DataFrame:
        '''
        self.normalize: str = < None | 'rpkm' | 'total' > 
        total = count * (total_count / sum(count))
        rpkm = count / (
            (plotted_width [bp] / 1e3 [bp]) *
            (sum(count) / 1e6 [reads])
        )
        '''
        self._validate_normalize_method()
        coverage_ = coverage.copy()
        total_count = coverage.attrs['total_counts']
        if total_count == 0:
            total_count = 1 #
            # this is to prevent a division by 0
            # setting to 1 will not change anything
            # as all counts are already 0
        if self.normalize == 'total':
            self._verify_total_count()
            scaling_factor = self.total_count / total_count
        if self.normalize == 'rpkm':
            scaling_factor = 1e9 / (self.region_width * total_count) #
            #  1e9 is generated by dividing the dividend by 1e3 (ip-K-m, kb)
            #   and 1e9 (ipk-M, 1e9 reads)
        coverage_['Count'] = coverage_['Count'] * scaling_factor
        return coverage_
    
    def normalize_coverage_grouped(self) -> None:
        '''
        Normalization based on "local" (on the selected
        region) total counts.
        In case also "size_factors" is specified, this 
        normalization will be ignored.
        '''
        if not self.normalize or self.size_factors is not None:
            return None
        coverages = map(self._normalize_coverage, self.coverage.values())
        names = self.coverage.keys()
        self.coverage = dict(zip(names, coverages))

    def _check_ylim(self) -> None:
        if not self.ylim:
            return None
        raise ValueError(
            '_check_ylim: ' +
            'Either "ylim_quantiles" or "ylim" can be specified'
        )
        
    def calculate_ylim_from_quantiles(self) -> None:
        if not self.ylim_quantiles:
            return None
        self._check_ylim()
        whole_coverage = [cov.Count.values for cov in self.coverage.values()]
        whole_coverage = np.array(whole_coverage).flatten()
        lower = np.quantile(whole_coverage, self.ylim_quantiles[0])
        upper = np.quantile(whole_coverage, self.ylim_quantiles[1])
        if upper == 0:
            # in case all tracks are 0, set the upper to 
            # 1 will allow the axis to be flipped in case 
            # the argument flip_minus_strand = True
            upper = 1
        self.ylim = (lower, upper)
    
    
    def plot_coverage(self, 
                      coverage: pd.DataFrame, 
                      ax: 'matplotlib.axes._subplots.AxesSubplot') -> None:
        color = coverage.attrs['color']
        ylabel = coverage.attrs['ylabel']
        
        ax.fill_between(
            coverage.Mid_bin, 
            np.zeros((coverage.shape[0], )),
            coverage.Count,
            color = color
        )

        if self.ylim:
            ax.set_ylim(*self.ylim)
        if self.flip_minus_strand and (coverage.attrs['Strand'] == '-'):
            ax.invert_yaxis()
        if self.show_ylabel:
            ax.set_ylabel(ylabel, **self.ylabel_kwargs)
        sns.despine(
            ax = ax, 
            trim = True, 
            offset = self.axes_offset
        )
            

    def generate_coverage_objects(self) -> None:
        # Specific for bulk pileup
        if not self.bgzf_files:
            return None
        self._validate_bin_size()
        self.create_ranges_bins()
        self.read_bgzf_files() # import file(s) as .bam
        if self.mode == 'tagmentation':
            self.generate_insertion_sites_grouped()
        self.set_bam_attrubutes_groued()
        self.integrate_metadata_grouped()
        self.split_bam_by_metadata()
        self.process_group_colors()
        self.set_group_colors_groued()
        self.calculate_total_counts()
        self.add_size_factors_attribute()
        if self.mode == 'pileup':
            self.split_bam_by_strand()
        self.compute_coverage_grouped()
        # Normalization and scales
        self.resize_coverage_grouped()
        self.normalize_coverage_grouped()
        self.calculate_ylim_from_quantiles()   
        # update plotting settings
        if self.mode == 'pileup':
            self.group_coverage_by_strand()
        self.plot_types_proportions_adjusted['coverage'] = (
            self.plot_types_proportions['coverage'] * 
            len(self.coverage)
        )
        self.selected_plots += ['coverage']
        self.axis_count += len(self.coverage)


class Strandness:
    def _validate_strandness(self) -> None:
        options = ['+', '-']
        is_valid = all([strand in options for strand in self.strandness])
        if is_valid:
            return None
        raise ValueError(
            f'_validate_strandness: strandness must be any of: ' +
            f'{options}. provided: {self.strandness}'
        )
        
    def _format_strandness(self) -> None:
        if isinstance(self.strandness, str):
            self.strandness = [self.strandness]
        if isinstance(self.strandness, tuple):
            self.strandness = list(self.strandness) 
        if isinstance(self.strandness, list):
            self._validate_strandness()
        
    def _filter_strand(self, name: str, bam: pd.DataFrame, strand: str) -> pd.DataFrame:
        bam = bam.query('Strand == @strand')
        bam.attrs['Strand'] = strand
        bam.attrs['ylabel'] = f'{name} | {strand}'
        if self.group_colors is None and isinstance(self.strand_colors, dict):
            bam.attrs['color'] = self.strand_colors[strand]
        return bam
    
    def split_bam_by_strand(self) -> None:
        if self.strandness is None:
            return None
        self._format_strandness()
        items = itertools.product(self.bam.items(), self.strandness)
        combinations = map(lambda item: itertools.chain(*item), items) # flatten ((1,2 ), 3) to (1, 2, 3)
        bams = list(map(self._filter_strand, *zip(*combinations)))
        names = list(map(lambda bam: bam.attrs['ylabel'], bams))
        self.bam = dict(zip(names, bams))

    def group_coverage_by_strand(self) -> None:
        if self.strandness is None:
            return None
        if len(self.strandness) == 1:
            return None
        if not self.groupby_strand:
            return None
        translate_map = str.maketrans('+-', '01')
        strands = list(map(lambda bam: bam.attrs['Strand'], self.bam.values()))
        codes = list(map(lambda strand: strand.translate(translate_map), strands))
        sorted_items = sorted(zip(codes, self.coverage.keys()))
        sorted_keys = list(map(lambda item: item[1], sorted_items))
        self.coverage = {key:self.coverage[key] for key in sorted_keys}


class PileupCoverage(Coverage, Strandness):
    def calculate_total_counts(self) -> None:
        for name, bam in self.bam.items():
            total_counts = np.sum(bam.End - bam.Start) / self.read_length
            bam.attrs['total_counts'] = total_counts


class TgmentationCoverage(Coverage):
    def generate_insertion_sites(self, bam: pd.DataFrame) -> pd.DataFrame:

        '''
        The tagmentation sites are single base.
        The reason why start and end are 1bp apart 
        is that the start of the range is considered
        closed by the pyranges coverage function (by
        default).
        Each fragment give rise to 2 independent tagmentation
        sites (at the very ends).
        '''
        # split the 2 tagmentations at each end of the fragment
        # into 2 data frames
        bam_l = bam.copy()
        bam_r = bam.copy()
        bam_l['End'] = bam_l.Start
        bam_l['Start'] = bam_l.Start - 1
        bam_r['Start'] = bam_r.End - 1
        # combine into a single data frame and add to the passed object
        bam = pd.concat([bam_l, bam_r], axis = 0)
        return bam
        

    def generate_insertion_sites_grouped(self) -> None:
        '''
        Modifies the object inplace by adding a bam 
        attribute.
        '''
        for name, bam in self.bam.items():
            self.bam[name] = self.generate_insertion_sites(bam)
      
    
    def calculate_total_counts(self) -> None:
        for name, bam in self.bam.items():
            bam.attrs['total_counts'] = len(bam)


class SingleCell:
    '''
    Compatible with multiple bed/fragments files
    however, in such case, if resizing by size_factors 
    is wanted, the size_factors will not match 
    the bam (dictionary) keys and therefore 
    this will return an exception.
    
    If needed to use the size_factors, either provide
    just one file and provide single cell size_factors
    or use multiple files and provide the bulk 
    size_factors.
    
    Both size_factors calculating functions are provided.
    '''
    def integrate_metadata(self, bam: pd.DataFrame) -> pd.DataFrame:
        assert self.adata is not None, (
            'integrate_metadata: adata object must be '+
            f'specified when setting: groupby = {self.groupby}'
        )
        self.metadata = self.adata.obs[self.groupby]

        attrs = bam.attrs.copy()
        
        bam = pd.DataFrame.merge(
            bam, 
            self.metadata, 
            how = 'inner', #1
            left_on = 'Barcode', 
            right_index = True
        )
        #1  has to be inner because the barcode must be in the metadata
        #   (some barcodes could have been removed because labeled as empty gems)

        bam.drop(columns = 'Barcode', inplace = True)

        bam.attrs = attrs
        return bam
    
    
    def integrate_metadata_grouped(self) -> None:
        if self.groupby is None:
            return None
        for name, bam in self.bam.items():
            self.bam[name] = self.integrate_metadata(bam)
            
            
    def split_bam_by_metadata(self) -> None:
        if self.groupby is None:
            return None
        bam = {}
        group_colors = {}
        for file_name, bam_ in self.bam.items():
            for group_name, group in bam_.groupby(self.groupby):
                if len(self.bam) > 1:
                    name = f'{file_name} | {group_name}'
                elif len(self.bam) == 1:
                    name = group_name
                bam[name] = group
                bam[name].attrs['ylabel'] = name
                if isinstance(self.group_colors, dict):
                    group_colors[name] = self.group_colors[group_name]
        self.bam = bam
        if isinstance(self.group_colors, dict):
            self.group_colors = group_colors

    def process_group_colors(self) -> None:
        if self.group_colors is None:
            return None
        if isinstance(self.group_colors, dict):
            return None  
        if isinstance(self.group_colors, str):
            group_colors = {}
            for name in self.bam.keys():
                group_colors[name] = self.group_colors
            self.group_colors = group_colors
        if isinstance(self.group_colors, tuple):
            self.group_colors = list(self.group_colors)
        if isinstance(self.group_colors, list):
            assert len(self.group_colors) == len(self.bam), (
                f'process_group_colors: group_colors {self.group_colors}' +
                f' and grops {self.bam} must have the same length'
            )
            items = zip(self.bam.keys(), self.group_colors)
            self.group_colors = {name: color for name, color in items}

    def set_group_colors_groued(self) -> None:
        if not isinstance(self.group_colors, dict):
            return None
        for name in self.bam.keys():
            self.bam[name].attrs['color'] = self.group_colors[name]


class PlotAssembly:
    # Figure setup methods
    def _reorder_plot_types(self) -> None:
        selected_plot = []
        for p in self.plots_order:
            if p in self.selected_plots:
                selected_plot.append(p)
        self.selected_plots = selected_plot

        
    def _compute_axis_heights(self) -> None:
        '''
        Adds axis_heights and axis_indexes attributes
        to the object.
        '''
        if not self.figsize:
            self.figsize = (9, self.axis_count * 1)

        cumulative_proportions = 0
        for k in self.selected_plots:
            cumulative_proportions += self.plot_types_proportions_adjusted[k]

        self.axis_heights = []
        self.axis_indexes = {}
        i = 0
        for k in self.selected_plots:
            axis_height =  self.figsize[1] * self.plot_types_proportions[k] 
            axis_height =  axis_height / cumulative_proportions
            if k == 'coverage':
                self.axis_heights += [axis_height] * len(self.coverage)
                self.axis_indexes[k] = [i + j for j in range(0, len(self.coverage))]
                i += len(self.coverage)
            else:   
                self.axis_heights += [axis_height]
                self.axis_indexes[k] = i
                i += 1
                
         
        
    def make_fugure(self) -> None: 
        if self.axis_count == 0:
            return None
        
        fig, ax = plt.subplots(
            len(self.axis_heights), 
            1, 
            figsize = self.figsize, 
            sharex = True,
            gridspec_kw = {'height_ratios':self.axis_heights},
            constrained_layout = True,
            squeeze = False
        )
        fig.set_constrained_layout_pads(
            **self.set_constrained_layout_pads_kwargs
        )

        for i in range(ax.shape[0]):
            ax.flat[i].set_xlim(self.start, self.end)
        
        for plot_type in self.selected_plots:
            if plot_type == 'links':
                self.plot_links(ax.flat[self.axis_indexes[plot_type]])
            
            elif plot_type == 'annotation':
                self.plot_annotation(ax.flat[self.axis_indexes[plot_type]])

            elif plot_type == 'peaks':
                self.plot_peaks(ax.flat[self.axis_indexes[plot_type]])

            elif plot_type == 'coverage':
                for i, (name, coverage) in enumerate(self.coverage.items()):
                    ax_ = ax.flat[self.axis_indexes[plot_type][i]]
                    ax_.grid(False)
                    self.plot_coverage(coverage, ax_)
                    
                    # Show y axis
                    if not self.y_axis_visible:
                        ax_.spines['left'].set_visible(False)
                        ax_.tick_params(left = False)
                        ax_.yaxis.set_ticklabels([])
                    
                    # show x axis only on the last plot
                    if i == len(self.coverage) - 1:
                        if not self.x_axis_visible:
                            ax_.spines['bottom'].set_visible(False)
                            ax_.tick_params(bottom = False)
                            ax_.xaxis.set_ticklabels([])
                        else:
                            ax_.set_xlabel(self.chr)
                    else:
                        ax_.spines['bottom'].set_visible(False)
                        ax_.tick_params(bottom = False)

        if self.suptitle:
            if isinstance(self.suptitle, str):
                fig.suptitle(
                    self.suptitle, 
                    x = 0.5, 
                    y = 1,
                    verticalalignment = 'bottom'
                )
            elif isinstance(self.suptitle, dict):
                fig.suptitle(**self.suptitle)
            
        if self.file_name:
            fig.savefig(
                self.file_name, 
                bbox_inches = "tight", 
                dpi = self.dpi
            )
            # set again constrained layout pads
            # (they reset after savefig is called)
            fig.set_constrained_layout_pads(
                **self.set_constrained_layout_pads_kwargs
            )

        # store the figure
        self.fig = fig


class Pileup(ObjectInitialization,
             Links, 
             Features, 
             PileupCoverage,
             SingleCell,
             ComponentsProcessing,
             PlotAssembly):
    '''
    Usage:
    -----------
    > 1 bgzf_files + groupby
        Multi sample, single cell mode 
        (the latter activated by groupby)
        All capabilities supported with exception 
        of size_factors correction
    > 1 bgzf_files
        Multi sample bulk mode
        If size_factors correction is desired, use
        the "compute_bulk_size_factors" function.
    1 bgzf_files
        Single sample bulk mode
    1 bgzf_files + groupby
        Single cell mode (the latter activated by groupby).
        If size_factors correction is desired, use
        the "compute_single_cell_size_factors" function.
    
    
    Info
    -----------
    Compositional inheritance from:
        ObjectInitialization, 
        Pileup,
        Links, 
        Features, 
        PileupCoverage
            Compositional inheritance from: 
                Coverage,
                Strandness
        SingleCell,
        PlotAssembly
    
    __init__ method provided by ObjectInitialization
    
    Needs to be used via the wrapper function "bulk_pileup" 
    as that function provides all default values. See docstring.
    '''


class Tagmentation(ObjectInitialization,
                   Links, 
                   Features, 
                   TgmentationCoverage,
                   SingleCell,
                   ComponentsProcessing,
                   PlotAssembly):
    '''
    Usage:
    -----------
    > 1 bgzf_files + groupby
        Multi sample, single cell mode 
        (the latter activated by groupby)
        All capabilities supported with exception 
        of size_factors correction
    > 1 bgzf_files
        Multi sample bulk mode
        If size_factors correction is desired, use
        the "compute_bulk_size_factors" function.
    1 bgzf_files
        Single sample bulk mode
    1 bgzf_files + groupby
        Single cell mode (the latter activated by groupby).
        If size_factors correction is desired, use
        the "compute_single_cell_size_factors" function.
    
    
    Info
    -----------
    Compositional inheritance from:
        ObjectInitialization, 
        Pileup,
        Links, 
        Features, 
        TgmentationCoverage
            Inherits from:
                Coverage, 
        SingleCell,
        PlotAssembly
    
    __init__ method provided by ObjectInitialization
    
    Needs to be used via the wrapper function "pileup" 
    as that function provides all default values. See docstring.
    '''

    
def _validate_mode(mode) -> None:
    options = ["pileup",  "tagmentation"]
    if mode in options:
        return None
    raise ValueError(
        f'_validate_mode: mode can be ' +
        f'one of: {options}. Provided: "{mode}"'
    )
    
def plot(region: Union[str, list[Union[str, int, int]], tuple, dict],
         extend_region: Union[int, dict[str, int], list[int], tuple[int]] = None,
         mode: str = 'pileup',
         # Annotation
         annotation: Union[pr.PyRanges, pd.DataFrame] = None,
         annotation_group_by: str = 'gene_name',
         annotation_main_feature: str = 'gene',
         annotation_line_color: str = 'grey',
         annotation_line_width: float = 0.5, # 0.25 with text
         annotation_height: float = 0.25,
         annotation_fill_color: str = 'lightgrey',
         annotation_arrow: bool = True,
         annotation_arrow_length: float = 0.01,
         annotation_names: Union[bool, tuple, list, str] = False,
         annotation_names_y_spacing: float = 0.05,
         annotation_clip: bool = False,
         # Peaks
         peaks: Union[bool, pr.PyRanges, pd.DataFrame] = False,
         peaks_line_color = 'grey',
         peaks_line_width: float = 0.5,
         peaks_height: float = 1,
         peaks_fill_color = 'lightgrey',
         peaks_names: Union[bool, tuple, list, str] = False,
         peaks_names_y_spacing: float = 0.05,
         peaks_clip: bool = False,
         # Coverage
         bgzf_files: Union[dict, list, tuple, str] = None,
         bin_size: int = 5,
         read_length: int = 75,
         strandness: Union[str, list] = None,
         flip_minus_strand: bool = True,
         groupby_strand: bool = False,
         strand_colors: Union[str, tuple, list, dict] = None,
         normalize: str = None,
         size_factors: dict = None,
         total_count: float = None,
         bgzf_files_colors: Union[str, tuple, list, dict] = 'grey',
         adata: anndata.AnnData = None,
         groupby: str = None,
         group_colors: Union[str, tuple, list, dict] = None,
         # Links
         links: Union[pd.DataFrame, pr.PyRanges] = None,
         links_orientation: str = 'bottom',
         links_color: str = 'grey',
         links_alpha: Union[float, int] = None,
         links_scale: Union[tuple, list] = (0, 1),
         links_linewidth: float = None,
         links_linewidth_multiplier: float = 1,
         links_overflowing_method: str = None,
         links_proportional_height: bool = False,
         links_min_height: float = 0.5,
         # Figure
         figsize: Union[tuple, list] = None,
         keep_empty_axis: bool = True,
         plot_types_proportions: dict[str, Union[int, float]] = {
             'annotation': 1.5,
             'peaks': 0.3,
             'coverage': 1,
             'links': 1
         },
         plots_order: Union[tuple[str], list[str]] = [
             'annotation',
             'peaks',
             'coverage',
             'links'
         ],
         file_name: str = None,
         dpi: int = 480,
         show_ylabel: bool = True,
         ylim: Union[tuple[Union[int, float]], list[Union[int, float]]] = None,
         ylim_quantiles: Union[tuple[Union[int, float]], list[Union[int, float]]] = None,
         y_axis_visible: bool = False,
         x_axis_visible: bool = False,
         ylabel_kwargs: dict[str, Union[str, int]] = {
             'rotation': 'horizontal',
             'labelpad': 20,
             'verticalalignment': 'center',
             'horizontalalignment': 'left'
         },
         set_constrained_layout_pads_kwargs: dict[str, Union[int, float]] = {
             'w_pad': 0,
             'h_pad': 0,
             'hspace': 0.05,
             'wspace': 0
         },
         axes_offset: Union[int, float] = 5,
         suptitle: Union[str, dict] = None,
         return_fig: bool = False,
         return_object: bool = False):

    '''
    Classes arguments
    ---------
    region: str | list | tuple | dict
        str: 'chr1:1-10' or 'chr1:1.1e3-10.2e4' or 'chr1_1_10'
        list: | tuple: ['chr1', 10, 1000]
        dict: {'chr': 'chr1', 'start': 10, 'end': 1000}
    extend_region: int | tuple | list | dict = None
        int: 20_000
        The value is extended from both start and end.
        tuple | list: [10_000, 30_000]
        The first element of the tuple will be subtracted from 
        the start and the second element will be added to the end.
        dict: {"start": 10_000, "end": 30_000} 
        extend the region of "extend_region" bp.
    mode = "pileup"
        Accepted values: "pileup" | "tagmentation"
    # Annotation
    annotation: pr.PyRanges | pd.DataFrame = None
        Pyranges (DataFrame) object with the following mandatory
        columns: Chromosome, Feature, Start, End, Strand
        and one or more of a grouping column, commonly named:
        gene_id, gene_name (gene_name is the default grouping column).
        It is suggested to import the annotation (from a gtf file)
        using the function read_gtf provided in the module, which
        formats the object accordingly.
    annotation_group_by: str = 'gene_name'
        Name of the column to be used to gather the exons. If the
        gtf file was imported with the read_gtf function this column
        will be present in the object.
    annotation_main_feature: str = 'gene'
        Should  be set to 'gene'.
    annotation_line_color: str = 'grey'
    annotation_line_width: float = 0.5
        Set it to 0.25 when using the  annotation_names option.
    annotation_height: float = 0.5
        as fraction of the y-axis.
        If annotation_names is set to True, the suggested value
        for this argument is 0.25 (to ensure enough space for the
        labels).
        The annotation is split in 2 rows: + strand above
        and - strand below, so, 0.5 means use the whole axis.
        If annotation_names is set to True, consider reducing
        this value to accomodate the labels.
    annotation_fill_color: str = 'lightgrey'
        Color of the inside of the exons.
    annotation_arrow: bool = True
        Whether to add the arrow sign marking the beginning of
        the genes and their orientation.
    annotation_arrow_length: float = 0.015
        As fraction of the x-axis.
        It appears only if 'annotation_names' is set to True
    annotation_names: bool | tuple | list | string = False
        It works only in case "annotation_arrow" is set to True (the
        labeling is hard to interpret, specially when the selected
        region is densely populated with genes if the beginning of
        the gene is not labeled with the arrow).
        If bool: Whether to add the gene/transcript (from annotation_group_by)
        to the plot.
        If provided by the user (tuple or list or string object):
        only the names provided will be added to the plot.
        It is case insensitive.
    annotation_names_y_spacing: float = 0.05
        The (absolute) proportion of the y-axis to leave as gap between the
        upper or lower bound of the annotation and the text.
    annotation_clip: bool = False
        If True, the range used is strictly the one provided.
        If False, then whenever a range in the annotation overlaps
        with either ends of the specified range, the range
        is extended until the end of that end. Setting this to True
        will give better representatios in most cases since the
        genes/transcripts are not truncated.

    # Peaks
    peaks: bool | pr.PyRanges | pd.DataFrame = False
        If set to True: the peaks object will be generated from
        the index of adata.var (peak names correspond to the
        coordinated), which makes it applicable only for a
        genomics (ATAC) adata object.
        If provided by the user (pr.PyRanges or pd.DataFrame
        object): a bed like data frame with mandatory columns:
        Chromosome, Start, End
        and an optional names column in case peaks_names
        is set to True
    peaks_line_color: str = 'grey'
    peaks_line_width: float = 0.5
    peaks_height: float = 1
        As fraction of the y-axis.
        If peaks_names is set to True, the suggested value
        for this argument is 0.6-0.7 depending on the size of the font
        (to ensure enough space for the labels).
    peaks_fill_color: str = 'lightgrey'
        Color on the inside of the peaks boxes.
    peaks_names: bool or tuple or list or string = False
        If bool: Whether to add the peak names to the plot.
        If provided by the user (tuple or list or string object):
        only the names provided will be added to the plot.
        It is case insensitive.
    peaks_names_y_spacing: float = 0.05
        The proportion of the y-axis to leave as gap between the
        upper bound of the peak and the text.
    peaks_clip: bool = False
        Same concept as for "annotation_clip" described above.

    # Coverage
    bgzf_files: dict | list | tuple | str = None
        Path to the fragments file.
        Normally this is generated automatically by
        cellranger count/aggregate.
        The file is a (gzipped) tabix file.
        The file is imported with pysam.TabixFile.
        The coverage represents the count of reads within a
        given bin. 1 read corresponds to 1 end of 1 fragment
        (therefore, each fragment produces 2 reads).
    read_length: int = 75
    bin_size: int = 5
        The size (in bp) of the bins used to aggregate the fragments.
    strandness: str | list = None 
        Accepted values: < None, "+", "-", ["+", "-"], ["-", "+"] >
        None = unstranded,
        "+" | "-" =  only +/- strand
        ["+", "-"] | ["-", "+"] = + and - | - and +
    flip_minus_strand: bool = True
        Used only in conjuction with strandness = ["+", "-"]
        Whether the y axis for the "-" strand should be flipped
        upside-down.
    groupby_strand: bool = False
        Used only in conjuction with strandness = ["+", "-"]
        Whether the coverage axes should be grouped by "strand"
        rather than by file name.
    strand_colors: str | tuple | list | dict = None
        If dict: {"+": col +, "-": col -}
        Lower precedence over the 3 color options
    bgzf_files_colors: str | tuple | list | dict =  'grey'
        Precedence over strand_colors but not over group_colors
    normalize: str = None
        Accepted values: < None, 'rpkm', 'total' >
        total = count * (total_count / sum(count))
        rpkm = count / (
            (plotted_width [bp] / 1e3 [bp]) *
            (sum(count) [reads] / 1e6 [reads])
        )
        Whether the read count must be normalized.
        No normalization if this argument is set to None.
        This option normally improves the visualization in particular
        when the groupby argument is specified to aggragate the counts.
        In such case, differences in accessibility between groups,
        or simply differences in cell count can give rise to
        different scales of magnitude.
        rpkm stands for reads per kilobase per million of 2
        reads.
        Since the size of the bam (with the real total insetionns)
        cannot be known (a range based fraction of the bam file
        is imported), it is extrapolated from the total read count
        on the specified region.
    size_factors: dict = None
        Meant to be used only in conjunction with groupby.
        Whether to use user provided size factors to scale each coverage
        track.
        read count values for each aggregate are brought to
        a common scale by dividing by the corresponding size factor.
        They can be computed with the function compute_size_factors.
        Or alternatively, one way to calculate them is by aggregating
        the fragment count (and then computing the ratios) from a
        metadata field if available on the particular adata object.
    total_count: float = None
        Must be set if normalize is set to 'total'.
    adata: anndata.AnnData = None
        anndata object to be used to merge the metadata.
        The object should be formatted in a typical scanpy manner
        namely, with the cell barcode provided as a single column 
        index (adata.obs.index).
        If multiple samples are being displayed, the user needs to 
        ensure that the cell barcodes in the bgzf files match the 
        ones found in the adata.obs.index. 
        One strategy to achieve this is to generate the adata object
        by concatenating multiple adata objects (one for each sample),
        this will add "-0, -1, ..." suffixes to the cell barcodes 
        (adata.obs.index). When generating the bgzf files, the same 
        suffixes (-0, -1, ...) need to be added to the cell barcodes 
        (matching the order of the adata objects).
        The companion scripts to generate the bgzf files are provided
        in the "bgzf_files_generation" directory.
    groupby: str = None
        Valid colum names of adata.obs. They will be integrated
        into the fragments object and used to aggragate the
        tagmentation count.
    group_colors: str | tuple | list | dict = None
        The color of the coverage tracks for each inidvidual group.
        Precedence over the other 2 color options

    # Links
    links: pd.DataFrame | pr.PyRanges = None
        If DataFrame: must contain columns: Chromosome, Start, End
        It can contain additional columns e.g. the one used by
        "links_alpha" and "links_linewidth"
    links_orientation: str = 'bottom'
        Accepted values: "top" | "bottom"
    links_color: str = 'grey'
    links_alpha: float | int | str = None
        If float or int: must be between 0 and 1.
        If str: it must be a column name of the links data frame
        In this case the valus will be interpolated with the "links_scale"
        and will be used as alpha value of the links
    links_scale: tuple | list = (0, 1)
        Used only if the argument links_alpha and/or links_linewidth
        are provided as a column of the links data frame.
        The min and max values that links_alpha and links_linewidth
        can take.
        If the column of the data frame to be used as links_alpha and/or
        links_linewidth is a correlation value, this value should be
        between 0 and 1.
    links_linewidth: float = None
        Same as links_alpha.
        Comment: if wanted to increase from the max value of 1,
        specify a higher value of "links_linewidth_multiplier".
    links_linewidth_multiplier: float = 1
        The amount to scale "links_linewidth"
    links_overflowing_method: str = None
        Accepted values: None | "extend_region" | "remove_overflowing"
        If None: the region won't be extended and the links won't
        be removed (simply, the portion of links overflowing the
        specified region will be cut).
        If "extend_region": the specified region will be extended to
        accomodate all links (including overflowing parts).
        Comment: this extension has precedence over the extension
        created by "annotation_clip" or "peaks_clip".
        If "remove_overflowing": the links that have an overflowing
        part will be completely removed.
    links_proportional_height: bool = False
        Whether the height of the links should reflect the relative
        link width.
    links_min_height: float = 0.5
        If "links_proportional_height" is set to True, this amount
        will be the proportional height of the shortest link.

    # Figure
    figsize: tuple | list = None
        If not specified (None), it is automatically set to
        (9, 1 * axis count)
    keep_empty_axis: bool = True
        Whetern to keep on the figure the annotation and peaks axis
        if they contain no plotted elements.
    plot_types_proportions: dict = {
        'annotation': 1.5,
        'peaks': 0.3,
        'coverage': 1
    }
        Values do not need to sum to 1, nor contain only the plot
        types that are selected.
    plots_order: tuple | list = [
        'annotation',
        'peaks',
        'coverage'
    ]
        The order of the plot types. It does not need to include only
        the selected plots in case not all types are selected.
    file_name: str = None
        Path to the file to save the figure.
        Not saved if it is set to 'None'.
    dpi: int = 480
    ylim: tuple | list = None
        Whether to force all the coverage y-axis to the same limit.
        Useful in conjuction with groupby and normalization.
    ylim_quantiles: tuple | list: = (0, 1)
        Mutually exclusive with "ylim".
        Set both "ylim_quantile" and "ylim" to None if it is
        desired to have independent ylim for each coverage axis.
        Set also "normalize" to None (default) if desired to
        plot the very raw read counts.
        The lower and upper quantiles are calculated on the
        combined coverage objects. (0, 1) = (total min, total max).
        This option is useful in case one or more groups presents
        outliers that need to be trimmed.
        One approach is to set the "ylim" manually, the other is
        to set this option with the upper bound lower than 1.
        The second option is preferred in case the multiple plots
        are generated in a loop.
    y_axis_visible: bool = False
        Whether to include in the coverage plot the y_axis.
    x_axis_visible: bool = False
        Whether to include in the coverage plot the x_axis.
    ylabel_kwargs: dict = {
        'rotation': 'horizontal',
        'labelpad': 20,
        'verticalalignment': 'center',
        'horizontalalignment': 'left'
    }
        Arguments passed to set_ylabel. The most important is likely
        the labelpad, to be increased to fit long names.
    show_ylabel: bool = True
    set_constrained_layout_pads_kwargs: dict = {
        'w_pad': 0,
        'h_pad': 0,
        'hspace': 0.05,
        'wspace': 0
    }
        Refer to matplotlib documentation for more details.
    axes_offset: int | float = 5
        Distance between the plot and the axis.
        It applies only to the lowest coverage plot.
    suptitle: str | dict = None
        If str: the title string
        If dict: the keyword arguments passed to Figure.suptitle.
    ------------

    Arguments specific for this function
    ------------
    return_fig: bool = False
        Whether to return the figure object.
        It has the precedence over the followint argument,
        so, in case both are set to true, the figure will
        be returned.
    return_object: bool = False
        Whether to return the BulkCoverage object
    ------------

    Returns
    ------------
    "matplotlib.figure.Figure" in case "return_fig" is set to True
    "BulkCoverage" in case "return_object" is set to True
    '''

    args = locals()
    _validate_mode(mode)
    if mode == 'pileup':
        c = Pileup(**args)
    if mode == 'tagmentation':
        c = Tagmentation(**args)
    c.generate_objects()
    c.make_fugure()

    if return_fig:
        # do not display the figure in Matplotlib
        # in case the figure should be returned
        plt.close()
        return c.fig

    if return_object:
        return c