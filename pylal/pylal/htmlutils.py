#!/usr/bin/env python

# =============================================================================
# Preamble
# =============================================================================

from __future__ import division
import re,socket
from glue import markup
from pylal import git_version

__author__  = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = "git id %s" % git_version.id
__date__    = git_version.date

"""
This module provides a few extensions to glue.markup to streamline GEO/LIGO detector characterisation tools that use very similar web interfaces
"""

# =============================================================================
# Write table
# =============================================================================

def write_table(headers, data, classdict={}):

    """
    Write table into glue.markup.page object. headers are written with <th>,
    multiple columns of data are written with <td>.

    Arguments:

        page : glue.markup.page
            page object into which to write table
        headers : list
            list of table header elements
        data : list
            list (or nested list) of table data elements, list of lists used for
            multiple rows

    Keyword arguments:

        classdict : dict
            dict containing tag:class pairs for table, td, th, and td HTML tags
            "table":"list" prints table with headers and data side-by-side,
            all others print all headers in one row, then all data in one row
            (use list of lists for multiple rows).
    """

    # extract classes
    tclass = classdict.get("table","")
    rclass = classdict.get("tr","")
    hclass = classdict.get("th","")
    dclass = classdict.get("td","")

    page = markup.page()

    # open table
    page.table(class_=tclass)

    # list: print two vertical columns of header:data pairs
    if tclass == "list":
         for i in range(len(headers)):
            page.tr(class_=rclass)
            page.th(str(headers[i]), class_=hclass)
            page.td(str(data[i]), class_=dclass)
            page.tr.close()

    # otherwise print "standard" table with single header row and multiple data
    # rows
    else:
        page.tr(class_=rclass)
        if len(headers)==1:
            page.th(str(headers[0]), colspan="100%", class_=hclass)
        else:
            for n in headers:
                page.th(str(n), class_=hclass)
        page.tr.close()

        if data and not re.search("list",str(type(data[0]))):
            data = [data]

        for row in data:
            page.tr(class_=rclass)
            for item in map(str, row):
                page.td(item, class_=dclass)

    page.table.close()

    return page

# =============================================================================
# Write <div id_="menubar">
# =============================================================================

def write_menu(sections, pages, current=None, classdict={"a":"menulink"}):

    """
    Returns glue.markup.page object for <div id_="menubar">, constructing menu
    in HTML.

    Arguments:

        sections : list
            ordered list of menu entry names
        pages : dict
            dict of section:href pairs holding link paths for each element of
            sections list

    Keyword arguments:

        current : str
            element of sections list to identify with class="open"
        classdict : dict
            dict of tag:class pairs for setting HTML tags

    """

    page = markup.page()
    page.div(id_="menubar", class_=classdict.get("div", ""))

    for i,sec in enumerate(sections):
        # set current class
        if sec == current and not re.search("open", classdict.get("a", "")):
           cl = classdict["a"] + " open"
        else:
           cl = classdict["a"]

        # remove index.html to make url look nicer
        if pages[sec].endswith("/index.html"):
            href = pages[sec][:-11]
        else:
            href = pages[sec]

        # make link
        page.a(sec, id_="a_%d" % i, class_=cl, href=href)

    page.div.close()

    return page

# =============================================================================
# Write glossary
# =============================================================================

def write_glossary(entries, htag="h1",\
                   classdict={"p":"line", "h4":"glossary closed"}):
    """
    Write a glossary of DQ terms into the glue.markup.page object page using the
    list of (term,definition) tuples terms.

    Arguments:

        entries : dict
            dict of term:definition pairs for inclusion in glossary

    Keyword arguments:

        htag : str
            HTML tag to use for header, default <h1>
        classdict : dict
            dict of tag:class pairs for HTML class assignment
    """

    page = markup.page()

    # write heading and description
    getattr(page, htag)("Glossary", id_="%s_glossary" % htag)
    page.div(id_="div_glossary", style="display: block;",\
             class_=classdict.get("div", ""))
    page.p("This section gives a glossary of terms relevant to this page.",\
           class_=classdict.get("p", ""))
    lvwiki = "https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/Acronyms"
    page.p("The LIGO-Virgo acronym wiki can be found on %s."\
           % markup.oneliner.a("this page", href=lvwiki),\
           class_=classdict.get("p", ""))

    # write glossary table
    terms = sorted(entries.keys())
    for i,term in enumerate(terms):
        page.h4(term, id_="glossaryh4_%d" % i, class_=classdict.get("h4", ""))
        page.div(entries[term], id_="div_%d" % i, style="display: none;",\
                 class_="glossary")

    page.div.close()

    return page

# =============================================================================
# Construct HTML base
# =============================================================================

def get_ldas_url():
    """
    Returns the url for http access to this host, based on its domain name
    Returns None-type if you"re not on a recognised LIGO-Virgo cluster.
    Should not be used when not on an LDAS cluster, can"t tell the difference 
    between nodes on CDS network at a site, and those on LDAS, for example...

    Example:
    >>> socket.getfqdn()
    ldas-pcdev1.ligo-la.caltech.edu
    >>> dqHTMLUtils.get_ldas_url()
    "https://ldas-jobs.ligo-la.caltech.edu"
    """

    # get fully qualified domain name (FQDN)
    fqdn = socket.getfqdn()

    # work out web root
    if re.search("ligo.caltech", fqdn):
        root = "https://ldas-jobs.ligo.caltech.edu"
    elif re.search("ligo-wa", fqdn):
        root = "https://ldas-jobs.ligo-wa.caltech.edu"
    elif re.search("ligo-la", fqdn):
        root = "https://ldas-jobs.ligo-la.caltech.edu"
    elif re.search("atlas", fqdn):
        match = re.search("atlas[13]", fqdn)
        if match:
            host = match.group()
            root = "https://%s.atlas.aei.uni-hannover.de" % host
        else:
            root = "https://atlas1.atlas.aei.uni-hannover.de"
    elif re.search("phy.syr", fqdn):
        root = "https://sugar-jobs.phy.syr.edu"
    elif re.search("astro.cf", fqdn):
        root = "https://gravity.astro.cf.ac.uk"
    elif re.search("phys.uwm", fqdn):
        root = "https://ldas-jobs.phys.uwm.edu"
    else:
        root = None

    return root

# =============================================================================
# Write simple page with plots and run information
# =============================================================================

def simple_page(header, htag="h1", desc=None, plotlist=None,
                args=None, version=None, classdict={}, init=False):
    """
    Returns a glue.markup.page object with an h1 heading, descriptive text,\
    some plots, and the arguments used to generate it.

    Designed for embedding/including as a frame in a larger page.

    Arguments:

        header : str
            text to write as header  (<h1>)
        htag : str
            HTML tag to use for header, default <h1>
        desc : str
            descriptive text to print below header(s) (<p>)
        plotlist : list
            list of filepath strings or (path, title) tuples to include
            (<a>,<img>)
        args : [ str | list ]
            string with arguments used to generate plots, or list of arguments
            to be space-separated
        version : str
            code version str/number to include
        classdict : dict
            dict containing HTML class strings for each tag used
        init : [ True | False ]
            initialise the markup.page object, adds HTML and BODY tags
    """

    # generate object
    page = markup.page()

    # initialise
    if init: page.init()

    # print header
    if header is not None:
        getattr(page, htag)(header, class_=classdict.get(htag,""),\
                            id_="%s_simple-page" % htag,
                            onclick="toggleVisible();")

    # set div
    hclass = classdict.get(htag, "open")
    if hclass == "closed":   display="none"
    else:                    display="block"
    page.div(class_=classdict.get("div",""), id_="div_simple-page",\
             style="display: %s;" % display)

    if desc is not None:
        page.p(desc, class_=classdict.get("p",""))

    # add plots
    if plotlist is not None:
        for p in plotlist:
            if isinstance(p, str):
                alt = None
            else:
                p,alt = p
            page.a(href=p, title=alt, class_=classdict.get("a",""))
            page.img(src=p, alt=alt, class_=classdict.get("img",""))
            page.a.close()

    if args is not None:
        page.p("Generated by running:", class_=classdict.get("p",""))
        if not isinstance(args, str):  args = " ".join(args)
        page.pre(args, class_=classdict.get("pre",""))
    if version is not None:
        page.p("Code version: %s" % version, class_=classdict.get("p",""))

    page.div.close()

    return page

# =============================================================================
# Construct page from various component
# =============================================================================

def build_page(icon=None, banner=None, homebutton=None, tabs=None,\
               menu=None, frame=None, **initargs):
    """
    Build a complete HTML page from 6 components: icon banner homebutton tabs
    menu frame. All other args passed to glue.markup.page.init function.
    See docstring of that function for help.

    Body is built to the following format:

    <div id_="container">
        <div class="content" id_="header">
            <div>
                <div class="nav">
                    ICON
                </div>
                <div class="frame">
                    BANNER
                </div>
            </div>
        </div>
        <div class="content" id_="tabs">
            <div>
                <div class="nav">
                    HOMEBUTTON
                </div>
                <div class="frame">
                    TABS
                </div>
            </div>
        </div>
        <div class="content" id_="main">
            <div>
                <div class="nav">
                    MENUBAR
                </div>
                <div class="frame">
                    FRAME
                </div>
            </div>
        </div>
    </div>
   
    """

    # setup page object
    page = markup.page()
    initargs.setdefault("doctype", "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">")
    page.init(**initargs)

    page.div(id_="container")

    # build top bar from icon and banner
    if icon is not None or banner is not None:
        page.div(class_="content", id_="header")
        page.div()
        page.div(str(icon), class_="nav", id_="headernav")
        page.div(str(banner), class_="frame", id_="headerframe")
        page.div.close()
        page.div.close()

    # build tab bar from home button and tab buttons
    if homebutton is not None or tabs is not None:
        page.div(class_="content", id_="tabs")
        page.div()
        if not homebutton: homebutton = ""
        page.div(str(homebutton), class_="nav", id_="tabsnav")
        page.div(str(tabs), class_="frame", id_="tabsframe")
        page.div.close()
        page.div.close()

    # build main page from menu and frame
    if menu is not None or frame is not None:
        page.div(class_="content", id_="main")
        page.div()
        page.div(str(menu), class_="nav", id_="mainnav")
        page.div(str(frame), class_="frame", id_="mainframe")
        page.div.close()
        page.div.close()

    page.div.close()

    return page

# =============================================================================
# Write summary page style page with plots and information
# =============================================================================

def summary_page(header=None, htag="h1", desc=None, plotlist=None,
                 text=None, subplots=None, info=None, classdict={},\
                 init=False):
    """
    Returns a glue.markup.page object with an heading, descriptive text,\
    summary section with one plot and text, section with other plots,
    section with subplots, then section with info.

    Designed for embedding/including as a frame in a larger page.

    Keyword arguments:

        header : str
            text to write as header  (<h1>)
        htag : str
            HTML tag to use for header, default <h1>
        desc : str
            descriptive text to print below header(s) (<p>)
        plotlist : list
            list of filepath strings or (path, title) tuples to include
            (<a>,<img>)
        text : [ str | glue.markup.page ]
            text to print below summary plot (<p>), or glue.markup.page to
            drop in with no enclosing tags.
        subplotlist : list
            list of filepath strings or (path, title) tuples to include
            (<a>,<img>)
        info : str
            information to print below summary plot (<p>), or
            glue.markup.page to drop in with no enclosing tags.
        classdict : dict
            dict containing HTML class strings for each tag used
        init : [ True | False ]
            initialise the markup.page object, adds HTML and BODY tags
    """

    # generate object
    page = markup.page()

    # initialise
    if init: page.init()

    # print header
    if header is not None:
        getattr(page, htag)(header, class_=classdict.get(htag,""),\
                            id_="%s_simple-page" % htag,
                            onclick="toggleVisible();")

    # set div
    hclass = classdict.get(htag, "open")
    if hclass == "closed":   display="none"
    else:                    display="block"
    page.div(class_=classdict.get("div"), id_="div_simple-page",\
             style="display: %s;" % display)

    # print description
    if desc is not None:
        page.p(desc, class_=classdict.get("p",""))

    # print first plot
    if plotlist is not None:
        Np = len(plotlist)
        if Np:
            p = plotlist[0]
            if isinstance(p, str):
                alt = None
            else:
                p,alt = p
            page.a(href=p, title=alt, class_=classdict.get("a",""))
            page.img(src=p, alt=alt, class_=classdict.get("img",""))
            page.a.close()

    # print text
    if text is not None:
        if isinstance(markup.page):
            page.add(text())
        else:
            page.p(str(text), class_=classdict.get("p",""))

    # add rest of plots
    if plotlist is not None and Np > 1:
        for p in plotlist[1:]:
            if isinstance(p, str):
                alt = None
            else:
                p,alt = p
            page.a(href=p, title=alt, class_=classdict.get("a",""))
            page.img(src=p, alt=alt, class_=classdict.get("img",""))
            page.a.close()

    if info is not None:
        if isinstance(markup.page):
            page.add(info())
        else:
            page.p(str(info), class_=classdict.get("p",""))

    page.div.close()

# =============================================================================
# Write about page
# =============================================================================

def about_page(executable, cmdargs, version=False, filedict={},\
               classdict={"h2":"open", "p":"line", "div":"about"}, init=False):
    """
    Returns a glue.markup.page object formatting the given executable,\
    commandline arguments, and any included files.

    Arguments:

        executable : string
            path of executable file (sys.argv[0])
        cmdargs : iterable
            set of command line arguments (sys.argv[1:])
 
    Keyword arguments:

        filedict : [ dict | iterable ]
            iterable of ("name", filepath) pairs to insert in full into the page
        classdict : dict
            dict containing HTML class strings for each tag used
        init : [ True | False ]
            initialise the markup.page object, adds HTML and BODY tags
    """
    page = markup.page()

    # initialise
    if init: page.init()

    page.h1("About", class_=classdict.get("h1",None))
    page.p("This page was generated with %s using the following tools."\
           % (executable), class_=classdict.get("p",None))

    def pre(h2, content, id_=0):
        page.h2(h2, id_="h2_%s" % id_, onclick="toggleVisible();",\
                class_=classdict.get("h2",None))
        page.div(id_="div_%s" % id_, style="display: block;",\
                 class_=classdict.get("div",None))
        page.pre(content, class_=classdict.get("pre",None))
        page.div.close()

    i = 0

    # write command line
    pre("Command line arguments", " ".join([executable]+cmdargs), id_=i)
    i += 1

    # write version
    if version:
        pre("Version", version, id_=i)
        i += 1
  
    if isinstance(filedict, dict):
        filedict = filedict.iteritems()

    for name,path in filedict:
        pre(name, open(path, "r").read(), id_=i)
        i += 1

    return page
