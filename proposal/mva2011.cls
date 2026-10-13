% mva2011.sty: MVA2011 style file for LaTeX
% mva2009.sty: MVA2009 style file for LaTeX,which is the same as
% modified by Ken-ichi MAEDA Aug 2008
% mva2002.sty: MVA2002 style file for LaTeX, which is the same as
% mva2000.sty: MVA2000 style file for LaTeX, which is the same as
% mva96.sty: MVA'96 style file for LaTeX
% Please use and modify at your own risk.
% Written by Takeshi Masuda tmasuda@etl.go.jp in May 1996.
% modified by Masaki Onishi, AIST, Oct 2011

%Usage:
%\documentclass{mva2011}
%\usepackage{other_packages}

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mva2011}[2010/10/13 v1.0 MVA2011 LaTeX class file]
\LoadClass[10pt,a4paper,twocolumn]{article}
\usepackage{balance}

\typeout{MVA Style File}
\pagestyle{empty}
\thispagestyle{empty}


\setlength{\voffset}{-1in}
\setlength{\topmargin}{21mm}
\setlength{\headheight}{0.0in}
\setlength{\headsep}{0.0in}
\setlength{\textheight}{255mm}

\setlength{\hoffset}{-1in}
\setlength{\oddsidemargin}{18mm}
\setlength{\marginparsep}{0pt}
\setlength{\textwidth}{174mm}
\setlength{\columnsep}{8mm}



%%\setlength{\itemsep}{0mm}
%\renewcommand{\itemsep}{0mm}


%%\setlength{\captionwidth}{10mm}
%%\setlength{\tabcolsep}{2mm}
%%\setlength{\itemsep}{0mm}
%\setlength{\parindent}{1em}


\def\section{\@startsection {section}{1}{\z@}{12pt plus 2pt minus 2pt}
{12pt plus 2pt minus 2pt}{\fontseries{b}\fontsize{12}{12}\selectfont}}
\def\subsection{\@startsection {subsection}{2}{\z@}{11pt plus 2pt minus 2pt}
{11pt plus 2pt minus 2pt}{\fontseries{b}\fontsize{11}{11}\selectfont}}
\def\subsubsection{\@startsection {subsubsection}{3}{\z@}{11pt plus 2pt minus 2pt}
{11pt plus 2pt minus 2pt}{\fontseries{b}\fontsize{11}{11}\selectfont}}


\renewcommand\normalsize{
\fontfamily{cmr}
\fontseries{m}
\fontshape{n}
\fontsize{10}{11}
\selectfont
}

\def\maketitle{\par
  \begingroup
  \def\thefootnote{\fnsymbol{footnote}}
  \def\@makefnmark{\hbox 
%  to 0pt{$^{\@thefnmark}$\hss}} 
% remove `to 0pt` for adjusting the space betweenfootnote mark and 
% footnote body by Yasumoto
  {$^{\@thefnmark}$\hss}} 
  \if@twocolumn 
  \twocolumn[\@maketitle] 
  \else \newpage
  \global\@topnum\z@ \@maketitle \fi
  \thispagestyle{empty}\@thanks
  \endgroup
  \setcounter{footnote}{0}
  \let\maketitle\relax
  \let\@maketitle\relax
  \gdef\@thanks{}\gdef\@author{}\gdef\@title{}\let\thanks\relax}

\def\@maketitle{\newpage
  \null
  \begin{center}
  {\fontseries{b}\fontsize{16}{18}\selectfont \@title \par}
  \vskip 2ex
  {
   \fontsize{12}{13}\selectfont
   \lineskip .5em
   \begin{tabular}[t]{c}\@author
   \end{tabular}\par
 }
 \vskip 1ex
 \end{center}
 \par
}

\def\thebibliography#1{\balance\section*{References}%
 \small\list{\@biblabel{\arabic{enumi}}}{
 \settowidth{\labelwidth}{\@biblabel{#1}}
 \setlength{\leftmargin}{\labelwidth}
 \addtolength{\leftmargin}{\labelsep}
 \partopsep=0pt \parsep=0pt \topsep=0pt \itemsep=0pt
 \usecounter{enumi}}
\def\newblock{\hskip .11em plus .33em minus -.07em}
\fussy
\sfcode`\.=1000\relax}

\flushbottom




\newlength{\@ctmp}
\newlength{\@figindent}
\setlength{\@figindent}{1pc}

\long\def\@makecaption#1#2{
  \vskip 10pt
  \setbox\@tempboxa\hbox{ #1.~#2}
  \setlength{\@ctmp}{\hsize}
  \addtolength{\@ctmp}{-\@figindent}\addtolength{\@ctmp}{-\@figindent}
  % IF longer than one indented paragraph line
  ~
  \ifdim \wd\@tempboxa >\@ctmp
  % THEN set as an indented paragraph
  \vspace*{-10mm}
  \begin{list}{}{\leftmargin\@figindent \rightmargin\leftmargin}
  \item[]#1.~#2\par
  \end{list}
  \else
  \hbox to\hsize{\hfil\box\@tempboxa\hfil} 
  \fi}


