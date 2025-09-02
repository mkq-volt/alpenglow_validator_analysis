import pygal

class LineBar(pygal.Line, pygal.Bar):
    def __init__(self, config=None, **kwargs):
        super(LineBar, self).__init__(config=config, **kwargs)
        self.y_title_secondary = kwargs.get('y_title_secondary')
        self.plotas = kwargs.get('plotas', 'line')

    def _make_y_title(self):
        super(LineBar, self)._make_y_title()
        
        # Add secondary title
        if self.y_title_secondary:
            yc = self.margin_box.top + self.view.height / 2
            xc = self.width - 10
            text2 = self.svg.node(
                self.nodes['title'], 'text', class_='title',
                x=xc,
                y=yc
            )
            text2.attrib['transform'] = "rotate(%d %f %f)" % (
                -90, xc, yc)
            text2.text = self.y_title_secondary

    def _plot(self):
        for i, serie in enumerate(self.series, 1):
            plottype = self.plotas

            raw_series_params = self.svg.graph.raw_series[serie.index][1]
            if 'plotas' in raw_series_params:
                plottype = raw_series_params['plotas']
                
            if plottype == 'bar':
                self.bar(serie)
            elif plottype == 'line':
                self.line(serie)
            else:
                raise ValueError('Unknown plottype for %s: %s'%(serie.title, plottype))

        for i, serie in enumerate(self.secondary_series, 1):
            plottype = self.plotas

            raw_series_params = self.svg.graph.raw_series[serie.index][1]
            if 'plotas' in raw_series_params:
                plottype = raw_series_params['plotas']

            if plottype == 'bar':
                self.bar(serie, True)
            elif plottype == 'line':
                self.line(serie, True)
            else:
                raise ValueError('Unknown plottype for %s: %s'%(serie.title, plottype))