import corner


def plot_corner(samples, labels, filename, **kwargs):

    if not isinstance(samples, list):
        fig = corner.corner(samples, labels=labels, **kwargs)
    else:
        fig = corner.corner(samples[0], labels=labels, **kwargs)
        fig = corner.corner(samples[1], labels=labels, fig=fig, **kwargs)

    fig.savefig(filename)
    return fig
