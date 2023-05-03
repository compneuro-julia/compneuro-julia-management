\NeedsTeXFormat{pLaTeX2e}
\ProvidesClass{kkgB5-repub}
  [2008/06/17 v0.02]

\newif\if@cameraready \@camerareadyfalse
\DeclareOption{cameraready}{\@camerareadytrue}

\newif\if@Proof \@Prooffalse
\DeclareOption{Proof}{\@Prooftrue}

\newcounter{@paper}
%%\newif\if@landscape \@landscapefalse
\newcommand{\@ptsize}{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagetrue
\newif\if@openright
\newif\if@mainmatter \@mainmattertrue
\hour\time \divide\hour by 60\relax
 \@tempcnta\hour \multiply\@tempcnta 60\relax
 \minute\time \advance\minute-\@tempcnta
\newif\if@stysize \@stysizefalse
\newif\if@enablejfam \@enablejfamtrue
\DeclareOption{b5paper}{\setcounter{@paper}{2}%
  \setlength\paperheight {232mm}
  \setlength\paperwidth  {182mm}}
\DeclareOption{10pt}{\renewcommand{\@ptsize}{0}}
\DeclareOption{11pt}{\renewcommand{\@ptsize}{1}}
\DeclareOption{12pt}{\renewcommand{\@ptsize}{2}}
\DeclareOption{tombow}{%
  \tombowtrue \tombowdatetrue
  \setlength{\@tombowwidth}{.1mm}%
  \@bannertoken{%
     \jobname\space:\space\number\year/\number\month/\number\day
      (\number\hour:\number\minute)}
  \maketombowbox}
\DeclareOption{tombo}{%
  \tombowtrue \tombowdatefalse
  \setlength{\@tombowwidth}{.1mm}%
  \maketombowbox}
\DeclareOption{mentuke}{%
  \tombowtrue \tombowdatefalse
  \setlength{\@tombowwidth}{\z@}%
  \maketombowbox}
\DeclareOption{tate}{%
  \AtBeginDocument{\tate\message{《縦組モード》}%
                   \adjustbaseline}%
}
\DeclareOption{oneside}{\@twosidefalse}
\DeclareOption{twoside}{\@twosidetrue}
\DeclareOption{onecolumn}{\@twocolumnfalse}
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{titlepage}{\@titlepagetrue}
\DeclareOption{notitlepage}{\@titlepagefalse}

\DeclareOption{openright}{\@openrighttrue}
\DeclareOption{openany}{\@openrightfalse}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{%
  \AtEndOfPackage{%
   \renewcommand\@openbib@code{%
      \advance\leftmargin\bibindent
      \itemindent -\bibindent
      \listparindent \itemindent
      \parsep \z@
      }%
   \renewcommand\newblock{\par}}}
\DeclareOption{disablejfam}{\@enablejfamfalse}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\DeclareOption{final}{\setlength\overfullrule{0pt}}

\ExecuteOptions{b5paper,twoside,onecolumn,final,openright}
\ProcessOptions\relax

\if@cameraready
 \input kkgB5-repub.fd
\else
%% normalsize の和文文字を 9pt 相当に
\DeclareFontShape{JY1}{mc}{m}{n}{<-> s * [0.961] jis}{}
\DeclareFontShape{JY1}{mc}{bx}{n}{<-> s * [0.961] jisg}{}
\DeclareFontShape{JY1}{gt}{m}{n}{<-> s * [0.961] jisg}{}
\DeclareFontShape{JY1}{gt}{bx}{n}{<-> s * [0.961] jisg}{}
\fi

\newdimen\JQ
\JQ=1.081Q%
\newdimen\Jpt
\Jpt=1.081pt%


%% it
\DeclareFontShape{JY1}{mc}{m}{it}{<->ssub*mc/m/n}{}
\DeclareFontShape{JY1}{mc}{bx}{it}{<->ssub*gt/m/n}{}
\DeclareFontShape{JY1}{gt}{m}{it}{<->ssub*gt/m/n}{}
\DeclareFontShape{JY1}{gt}{bx}{it}{<->ssub*gt/m/n}{}
\DeclareFontShape{JT1}{mc}{m}{it}{<->ssub*mc/m/n}{}
\DeclareFontShape{JT1}{mc}{bx}{it}{<->ssub*gt/m/n}{}
\DeclareFontShape{JT1}{gt}{m}{it}{<->ssub*gt/m/n}{}
\DeclareFontShape{JT1}{gt}{bx}{it}{<->ssub*gt/m/n}{}
%% sl
\DeclareFontShape{JY1}{mc}{m}{sl}{<->ssub*mc/m/n}{}
\DeclareFontShape{JY1}{mc}{bx}{sl}{<->ssub*gt/m/n}{}
\DeclareFontShape{JY1}{gt}{m}{sl}{<->ssub*gt/m/n}{}
\DeclareFontShape{JY1}{gt}{bx}{sl}{<->ssub*gt/m/n}{}
\DeclareFontShape{JT1}{mc}{m}{sl}{<->ssub*mc/m/n}{}
\DeclareFontShape{JT1}{mc}{bx}{sl}{<->ssub*gt/m/n}{}
\DeclareFontShape{JT1}{gt}{m}{sl}{<->ssub*gt/m/n}{}
\DeclareFontShape{JT1}{gt}{bx}{sl}{<->ssub*gt/m/n}{}
%% sc
\DeclareFontShape{JY1}{mc}{m}{sc}{<->ssub*mc/m/n}{}
\DeclareFontShape{JY1}{mc}{bx}{sc}{<->ssub*gt/m/n}{}
\DeclareFontShape{JY1}{gt}{m}{sc}{<->ssub*gt/m/n}{}
\DeclareFontShape{JY1}{gt}{bx}{sc}{<->ssub*gt/m/n}{}
\DeclareFontShape{JT1}{mc}{m}{sc}{<->ssub*mc/m/n}{}
\DeclareFontShape{JT1}{mc}{bx}{sc}{<->ssub*gt/m/n}{}
\DeclareFontShape{JT1}{gt}{m}{sc}{<->ssub*gt/m/n}{}
\DeclareFontShape{JT1}{gt}{bx}{sc}{<->ssub*gt/m/n}{}

\renewcommand{\normalsize}{\@setfontsize\normalsize\@xpt{22H}%
 \abovedisplayskip 10\p@ \@plus2\p@ \@minus5\p@
 \abovedisplayshortskip \z@ \@plus3\p@
 \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
 \belowdisplayskip \abovedisplayskip
 \let\@listi\@listI}

\renewcommand{\normalsize}{\fontsize{12.5\JQ}{22H}\selectfont}

\normalsize

\setbox0\hbox{\char\euc"A1A1}% "
\setlength\Cht{\ht0}
\setlength\Cdp{\dp0}
\setlength\Cwd{\wd0}
\setlength\Cvs{\baselineskip}
\setlength\Chs{\wd0}

\newcommand{\small}{\@setfontsize\small\@ixpt{11}%
 \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
 \abovedisplayshortskip \z@ \@plus2\p@
 \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
 \def\@listi{\leftmargin\leftmargini
             \topsep 4\p@ \@plus2\p@ \@minus2\p@
             \parsep 2\p@ \@plus\p@ \@minus\p@
             \itemsep \parsep}%
 \belowdisplayskip \abovedisplayskip}

\newcommand{\footnotesize}{\@setfontsize\footnotesize\@viiipt{10}%
 \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
 \abovedisplayshortskip \z@ \@plus\p@
 \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
 \def\@listi{\leftmargin\leftmargini
             \topsep 3\p@ \@plus\p@ \@minus\p@
             \parsep 2\p@ \@plus\p@ \@minus\p@
             \itemsep \parsep}%
 \belowdisplayskip \abovedisplayskip}

\newcommand{\scriptsize}{\@setfontsize\scriptsize\@viipt\@viiipt}
\newcommand{\tiny}{\@setfontsize\tiny\@vpt\@vipt}
\newcommand{\large}{\@setfontsize\large\@xiipt{19}}
\newcommand{\Large}{\@setfontsize\Large\@xivpt{23}}
\newcommand{\LARGE}{\@setfontsize\LARGE\@xviipt{25}}
\newcommand{\huge}{\@setfontsize\huge\@xxpt{28}}
\newcommand{\Huge}{\@setfontsize\Huge\@xxvpt{33}}

\hoffset -.45in
\headheight 12\p@
\headsep 5mm
\topskip 1\Cht
\footskip 0mm%12mm
\maxdepth .5\topskip
\textwidth45zw
\textheight 36\Cvs
\advance \textheight \topskip
\advance \textheight .05mm

\if@twocolumn
  \setlength\marginparsep{24\p@}
\else
  \setlength\marginparsep{24\p@}
\fi
\setlength\marginparpush{5\p@}

\iftombow
% \topmargin\paperheight
% \advance \topmargin -\textheight
% \advance \topmargin -\headheight
% \advance \topmargin -\headsep
% \advance \topmargin -\footskip
% \divide \topmargin \tw@
 %%\advance \topmargin mm
 \advance \topmargin -1in
 \advance \topmargin 6mm%13.5mm
 %\@settopoint\topmargin
 \oddsidemargin\paperwidth
 \advance \oddsidemargin -\textwidth
 \divide \oddsidemargin \tw@
 \advance \oddsidemargin  -1in
 \evensidemargin\oddsidemargin
 \advance \oddsidemargin 1.5mm %%%
 \advance \evensidemargin -1.5mm %%%
\else
 \topmargin \z@
 %\@settopoint\topmargin
 \oddsidemargin\z@
 %\@settopoint\oddsidemargin
 \evensidemargin\z@
 %\@settopoint\evensidemargin
\fi

\marginparwidth 20mm
%\@settopoint\marginparwidth

\footnotesep 7pt
\skip\footins .65\Cvs \@plus .25\Cvs \@minus .15\Cvs
\floatsep 7mm \@plus 2\p@ \@minus 2\p@
\dblfloatsep 7mm \@plus 2\p@ \@minus 2\p@
\textfloatsep 7mm \@plus 2\p@ \@minus 4\p@
\dbltextfloatsep 7mm \@plus 2\p@ \@minus 4\p@
\intextsep 7mm \@plus 2\p@ \@minus 2\p@

\@fptop 0\p@ \@plus 1fil
\@fpsep 8\p@ \@plus 2fil
\@fpbot 0\p@ \@plus 1fil
\@dblfptop 0\p@ \@plus 1fil
\@dblfpsep 8\p@ \@plus 2fil
\@dblfpbot 0\p@ \@plus 1fil
\partopsep\z@

\def\@listi{\leftmargin\leftmargini
  \parsep  4\p@ \@plus2\p@ \@minus\p@
  \topsep  8\p@ \@plus2\p@ \@minus4\p@
  \itemsep 4\p@ \@plus2\p@ \@minus\p@
}
\let\@listI\@listi
\@listi
\def\@listii{\leftmargin\leftmarginii
   \labelwidth\leftmarginii \advance\labelwidth-\labelsep
   \topsep 4\p@ \@plus2\p@ \@minus\p@
   \parsep 2\p@ \@plus\p@  \@minus\p@
   \itemsep\parsep}
\def\@listiii{\leftmargin\leftmarginiii
   \labelwidth\leftmarginiii \advance\labelwidth-\labelsep
   \topsep   2\p@ \@plus\p@\@minus\p@
   \parsep   \z@
   \partopsep\p@ \@plus\z@ \@minus\p@
   \itemsep\topsep}
\def\@listiv {\leftmargin\leftmarginiv
              \labelwidth\leftmarginiv
              \advance\labelwidth-\labelsep}
\def\@listv  {\leftmargin\leftmarginv
              \labelwidth\leftmarginv
              \advance\labelwidth-\labelsep}
\def\@listvi {\leftmargin\leftmarginvi
              \labelwidth\leftmarginvi
              \advance\labelwidth-\labelsep}

\columnsep 2zw
\columnseprule 0\p@
\lineskip 1\p@ 
\normallineskip 1\p@
\renewcommand{\baselinestretch}{}
\parskip\z@
\parindent1zw

\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{5}% 2
\setcounter{bottomnumber}{4}% 1
\setcounter{totalnumber}{9}% 3
\setcounter{dbltopnumber}{5}% 2
\renewcommand{\topfraction}{.95}% .7
\renewcommand{\bottomfraction}{.95}% .3
\renewcommand{\textfraction}{.05}% .2
\renewcommand{\floatpagefraction}{.05}% .5
\renewcommand{\dbltopfraction}{.95}% .7
\renewcommand{\dblfloatpagefraction}{.05}% .5

\if@Proof
 \def\@Rtop{\rlap{\smash{\rule[-\headsep]{0.05mm}{10mm}%
  \rule[-\headsep]{10mm}{0.05mm}}}}
 \def\@Ltop{\llap{\smash{\rule[-\headsep]{10mm}{0.05mm}%
  \rule[-\headsep]{0.05mm}{10mm}}}}
 \def\@Rbot{\rlap{\smash{\raisebox{1\footskip}
  {\rule[-10mm]{0.05mm}{10mm}\rule{10mm}{0.05mm}}}}}
 \def\@Lbot{\llap{\smash{\raisebox{1\footskip}
  {\rule{10mm}{0.05mm}\rule[-10mm]{0.05mm}{10mm}}}}}
\else
 \let\@Rtop\relax\let\@Ltop\relax\let\@Rbot\relax\let\@Lbot\relax
\fi

\def\ps@empty{\let\ps@jpl@in\ps@plain%
 \def\@oddhead{\@Ltop\hfil\@Rtop}%
 \let\@evenhead\@oddhead
 \def\@oddfoot{\@Lbot\hfil\@Rbot}%
 \let\@evenfoot\@oddfoot
 \let\@mkboth\@gobbletwo
 \let\chaptermark\@gobble
 \let\sectionmark\@gobble
}

\def\ps@plain{\let\@mkboth\@gobbletwo
 \let\ps@jpl@in\ps@plain
 \def\@oddhead{\@Ltop\hfil{\nombre@font\thepage}\@Rtop}%
 \def\@evenhead{\@Ltop{\nombre@font\thepage}\hfil\@Rtop}%
 \def\@oddfoot{\@Lbot\hfil\@Rbot}%
 \let\@evenfoot\@oddfoot
}
\let\ps@jpl@in\ps@plain

\def\ps@headnombre{\let\@mkboth\@gobbletwo
 \let\ps@jpl@in\ps@headnombre
 \def\@oddhead{\@Ltop\hfil{\nombre@font\thepage}\@Rtop}%
 \def\@evenhead{\@Ltop{\nombre@font\thepage}\hfil\@Rtop}%
 \def\@oddfoot{\@Lbot\hfil\@Rbot}%
 \let\@evenfoot\@oddfoot
}

\def\ps@footnombre{\let\@mkboth\@gobbletwo
 \let\ps@jpl@in\ps@footnombre
 \def\@oddfoot{\@Lbot\hfil{\nombre@font\thepage}\@Rbot}%
 \def\@evenfoot{\@Lbot{\nombre@font\thepage}\hfil\@Rbot}%
 \def\@oddhead{\@Ltop\hfil\@Rtop}%
 \let\@evenhead\@oddhead
}

\def\head@font{\small\normalfont}
\def\nombre@font{\small\slshape}


\def\ps@headings{\let\ps@jpl@in\ps@headnombre
 \def\@oddhead{\@Ltop\hfil{\head@font\rightmark}\hskip2zw
  {\nombre@font\thepage}}%
 \def\@evenhead{{\nombre@font\thepage}\hskip2zw
  {\head@font\leftmark}\hfil\@Rtop}%
 \def\@oddfoot{\@Lbot\hfil\@Rbot}%
 \let\@evenfoot\@oddfoot
 \let\@mkboth\markboth
 \def\chaptermark##1{\markboth{%
  \ifnum \c@secnumdepth >\m@ne
   \if@mainmatter
    \@chapapp\thechapter\@chappos\hskip1zw
   \fi
  \fi
   \inhibitglue ##1\inhibitglue}{}}%
 \def\sectionmark##1{\markright{%
  \ifnum \c@secnumdepth >\z@ \thesection\hskip1zw\fi
   \inhibitglue ##1\inhibitglue}}%
}

\def\ps@bothstyle{\let\ps@jpl@in\ps@footnombre
 \def\@oddhead{\@Ltop\hfil{\head@font\rightmark}\@Rtop}%
 \def\@evenhead{\@Ltop{\head@font\leftmark}\hfil\@Rtop}%
 \def\@oddfoot{\@Lbot\hfil{\nombre@font\thepage}\@Rbot}%
 \def\@evenfoot{\@Lbot{\nombre@font\thepage}\hfil\@Rbot}%
 \let\@mkboth\markboth
 \def\chaptermark##1{\markboth{%
    \ifnum \c@secnumdepth >\m@ne
	\if@mainmatter
	\@chapapp\thechapter\@chappos\hskip1zw
	\fi
    \fi
      ##1}{}}%
 \def\sectionmark##1{\markright{%
   \ifnum \c@secnumdepth >\z@ \thesection\hskip1zw\fi
    ##1}}%
}

\def\ps@myheadings{\let\ps@jpl@in\ps@plain%
 \def\@oddhead{\@Ltop\hfil{\head@font\rightmark}\hskip2zw
  {\nombre@font\thepage}\@Rtop}%
 \def\@evenhead{\@Ltop{\nombre@font\thepage}\hskip2zw
  {\head@font\leftmark}\hfil\@Rtop}%
 \def\@oddfoot{\@Lbot\hfil\@Rbot}%
 \let\@evenfoot\@oddfoot
 \let\@mkboth\@gobbletwo
 \let\chaptermark\@gobble
 \let\sectionmark\@gobble
}

\if@titlepage
  \newcommand{\maketitle}{\begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \let\footnote\thanks
  \null\vfil
  \vskip 60\p@
  \begin{center}%
    {\LARGE \@title \par}%
    \vskip 3em%
    {\Large
     \lineskip .75em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
      \vskip 1.5em%
    {\large \@date \par}%       % Set date in \large size.
  \end{center}\par
  \@thanks\vfil\null
  \end{titlepage}%
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\p@thanks\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
  }%
\else
  \newcommand{\maketitle}{\par
  \begingroup
    \renewcommand{\thefootnote}{\fnsymbol{footnote}}%
    \def\@makefnmark{\hbox{\ifydir $\m@th^{\@thefnmark}$
      \else\hbox{\yoko$\m@th^{\@thefnmark}$}\fi}}%
     \long\def\@makefntext##1{\parindent 1em\noindent
       \hbox to1.8em{\hss$\m@th^{\@thefnmark}$}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne \@maketitle
      \else \twocolumn[\@maketitle]%
      \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \fi
     \thispagestyle{empty}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\p@thanks\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
  }
  \def\@maketitle{%
  \newpage\null
  \vskip 2em%
  \begin{center}%
  \let\footnote\thanks
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par\vskip 1.5em}
\fi

\newcommand*{\chaptermark}[1]{}
\setcounter{secnumdepth}{3}

\newcounter{part}
\newcounter{chapter}
\newcounter{section}[chapter]
\newcounter{subsection}[section]
\newcounter{subsubsection}[subsection]
\newcounter{paragraph}[subsubsection]
\newcounter{subparagraph}[paragraph]

\renewcommand{\thepart}{\@Roman\c@part}
\renewcommand{\thechapter}{\@arabic\c@chapter}
\renewcommand{\thesection}{\thechapter.\@arabic\c@section}
\renewcommand{\thesubsection}{\thesection.\@arabic\c@subsection}
\renewcommand{\thesubsubsection}{%
 \thesubsection.\@arabic\c@subsubsection}
\renewcommand{\theparagraph}{%
 \thesubsubsection.\@arabic\c@paragraph}
\renewcommand{\thesubparagraph}{%
 \theparagraph.\@arabic\c@subparagraph}

\newcommand{\@chapapp}{\prechaptername}
\newcommand{\@chappos}{\postchaptername}
\newcommand\frontmatter{%
  \if@openright \cleardoublepage \else \clearpage \fi
  \@mainmatterfalse\pagenumbering{roman}}
\newcommand{\mainmatter}{%
  \if@openright \cleardoublepage \else \clearpage \fi
  \@mainmattertrue\pagenumbering{arabic}}
\newcommand{\backmatter}{%
  \if@openright \cleardoublepage \else \clearpage \fi
  \@mainmatterfalse}

\if@Proof
 \def\cleardoublepage{\clearpage
  \if@twoside
   \ifodd\c@page
   \else
%    \clearpage
%    \addtocounter{page}{1}%
    \hbox{}\thispagestyle{empty}\newpage
    \if@twocolumn\hbox{}\newpage\fi
    \clearpage
   \fi
  \fi}
\else
 \def\cleardoublepage{\clearpage
  \if@twoside
   \ifodd\c@page
   \else
    \hbox{}\thispagestyle{empty}\newpage
    \if@twocolumn\hbox{}\newpage\fi
    \clearpage
   \fi
  \fi}
\fi

\def\part{% }{
  \if@openright \cleardoublepage \else \clearpage \fi
  \thispagestyle{empty}%
  \if@twocolumn\onecolumn\@tempswatrue\else\@tempswafalse\fi
  \null\vfil
  \secdef\@part\@spart}

\def\@part[#1]#2{%
  \ifnum \c@secnumdepth >-2\relax
    \refstepcounter{part}%
    \addcontentsline{toc}{part}{%
       \prepartname\thepart\postpartname\hspace{1em}#1}%
  \else
    \addcontentsline{toc}{part}{#1}%
  \fi
  \markboth{}{}%
  {\centering
   \interlinepenalty\@M\reset@font
   \ifnum \c@secnumdepth >-2\relax
     \huge\bfseries\prepartname\thepart\postpartname
     \par\vskip20\p@
   \fi
   \Huge\bfseries#2\par}%
   \@endpart}

\def\@spart#1{{%
  \centering
  \interlinepenalty\M@\reset@font
  \Huge\bfseries#1\par}%
  \@endpart}

\def\@endpart{\vfil\newpage
   \if@twoside\null\thispagestyle{empty}\newpage\fi
   \if@tempswa\twocolumn\fi}

\def\chapter{% }{
 \if@openright\cleardoublepage\else\clearpage\fi
 \thispagestyle{empty}%
 \global\@topnum\z@
 \@afterindenttrue
 \secdef\@chapter\@schapter}

\def\@chapter[#1]#2{%
 \ifnum \c@secnumdepth >\m@ne
  \if@mainmatter
   \refstepcounter{chapter}%
   \typeout{\@chapapp\space\the\c@chapter\space\@chappos}%
   \addcontentsline{toc}{chapter}%
    {\protect\numberline{\@chapapp\thechapter\@chappos}#1}%
  \else
   \addcontentsline{toc}{chapter}{#1}%
  \fi
 \else
  \addcontentsline{toc}{chapter}{#1}%
 \fi
 \chaptermark{#1}%
 %\addtocontents{lof}{\protect\addvspace{\protect\baselineskip}}%
 %\addtocontents{lot}{\protect\addvspace{\protect\baselineskip}}%
 \@makechapterhead{#2}%
 \@afterheading
}

\def\chap@font{\huge\bfseries}%%%%%

\def\@makechapterhead#1{\hbox{}%
\hrule width0pt \vspace{-1zw}
 \bgroup
  \parindent\z@
  \boxmaxdepth\z@
  \lineskiplimit\z@
  \lineskip\z@
  \vbox to 37mm{\vss
   \reset@font\chap@font
   \setbox\z@\hbox{\@chapapp\thechapter\@chappos\hskip1zw
    \shortstack{\inhibitglue #1\inhibitglue}}%
   \ifdim\wd\z@>\linewidth
    \setlength\@tempdima{\linewidth}%
    \setbox\z@\hbox{\@chapapp\thechapter\@chappos\hskip1zw}%
    \addtolength\@tempdima{-\wd\z@}%
    \unhbox\z@\nobreak
    \vtop{\hsize\@tempdima\inhibitglue #1\inhibitglue}%
   \else
    \centering
    \@chapapp\thechapter\@chappos\hskip1zw\nobreak
    \begin{tabular}[t]{@{}l@{}}
     \inhibitglue #1\inhibitglue
    \end{tabular}%
   \fi
  \vss}\par
 \egroup
}

\def\@schapter#1{%
 \@makeschapterhead{#1}%
 \@afterheading
}

\def\schap@font{\LARGE\bfseries}

\def\@makeschapterhead#1{\hbox{}%
 \bgroup
  \parindent\z@
  \boxmaxdepth\z@
  \lineskiplimit\z@
  \lineskip\z@
  \vbox to 6\Cvs{\vss%%%%%
   \schap@font #1%
  \vss}\par
 \egroup
}

\def\@afterheading{%
 \global\@nobreaktrue
 \everypar{%
  \if@nobreak
   \global\@nobreakfalse
   \clubpenalty \z@
    \if@afterindent
    \else
     {\setbox\z@=\lastbox}%
    \fi
  \else
   \clubpenalty\@clubpenalty
   \everypar{}%
  \fi}%
}

%% \@startsection{NAME}{LEVEL}{INDENT}{BEFORESKIP}{AFTERSKIP}{STYLE}
\def\@startsection#1#2#3#4#5#6{%
 \if@noskipsec \leavevmode \fi
 \par \@tempskipa #4\relax
 \@afterindenttrue
 \ifdim \@tempskipa <\z@ \@tempskipa -\@tempskipa \@afterindentfalse\fi
 \if@nobreak
  \everypar{}%
 \else
  %\addpenalty\@secpenalty
  \addvspace\@tempskipa
 \fi
 \@ifstar{\@ssect{#3}{#4}{#5}{#6}}{\@dblarg{\@sect{#1}{#2}{#3}{#4}{#5}{#6}}}}

%% \@sect{NAME}{LEVEL}{INDENT}{BEFORESKIP}{AFTERSKIP}{STYLE}[ARG1]{ARG2}
%% \ifdim \@tempskipa>\z@ --> \ifdim \@tempskipa<\z@
\def\@sect#1#2#3#4#5#6[#7]#8{%
 \ifnum #2>\c@secnumdepth
  \let\@svsec\@empty
 \else
  \refstepcounter{#1}%
  \protected@edef\@svsec{\@seccntformat{#1}\relax}%
 \fi
  \@tempskipa #5\relax
  \ifdim \@tempskipa<\z@
   \def\@svsechd{%
     #6{\hskip #3\relax
     \@svsec \inhibitglue #8\inhibitglue}%
   \csname #1mark\endcsname{#7}%
   \addcontentsline{toc}{#1}{%
     \ifnum #2>\c@secnumdepth \else
       \protect\numberline{\csname the#1\endcsname}%
     \fi
    #7}}%
  \else
   \begingroup
    #6{%
      \@hangfrom{\hskip #3\relax\@svsec}%
       \interlinepenalty\@M\inhibitglue #8\inhibitglue\@@par
      }%
   \endgroup
   \csname #1mark\endcsname{#7}%
   \addcontentsline{toc}{#1}{%
     \ifnum #2>\c@secnumdepth \else
      \protect\numberline{\csname the#1\endcsname}%
       \fi
    #7}%
  \fi
  \@xsect{#5}}

%% \@ssect{INDENT}{BEFORESKIP}{AFTERSKIP}{STYLE}{ARG} ==
%% \ifdim \@tempskipa>\z@ --> \ifdim \@tempskipa<\z@
\def\@ssect#1#2#3#4#5{%
  \@tempskipa #3\relax
  \ifdim \@tempskipa<\z@
   \def\@svsechd{#4{\hskip #1\relax #5}}%
  \else
    \begingroup
      #4{%
        \@hangfrom{\hskip #1}%
         \interlinepenalty\@M\inhibitglue #5\inhibitglue\@@par
        }%
    \endgroup
  \fi
  \@xsect{#3}}

%% \ifdim \@tempskipa>\z@ --> \ifdim \@tempskipa<\z@
\def\@xsect#1{%
  \@tempskipa #1\relax
  \ifdim \@tempskipa<\z@
    \@nobreakfalse
    \global\@noskipsectrue
    \everypar{%
      \if@noskipsec
        \global\@noskipsecfalse
       {\setbox\z@\lastbox}%
        \clubpenalty\@M
        \begingroup \@svsechd \endgroup
        \unskip
        \@tempskipa #1\relax
        \hskip -\@tempskipa
        \pagebreak[2]%
      \else
        \clubpenalty \@clubpenalty
        \everypar{}%
      \fi}%
  \else
    \par \nobreak
    \vskip \@tempskipa
    \@afterheading
  \fi
  \ignorespaces}

\def\@seccntformat#1{\csname the#1\endcsname\hskip1zw}

\def\secsubtitle{}

\def\sec@font{\fontsize{18\JQ}{22H}\selectfont\baselineskip\Cvs\rmhfamily}
\def\secnum@font{\fontsize{17Q}{22H}\selectfont\usefont{T1}{pad}{b}{n}}
\def\secsubtit@font{\fontsize{10\JQ}{22H}\selectfont\rmmfamily}
\def\subsec@font{\fontsize{14\JQ}{22H}\selectfont\mdgdbfamily}
\def\subsecnum@font{\fontsize{12Q}{22H}\selectfont\usefont{T1}{pad}{b}{n}}
\def\subsubsec@font{\fontsize{13.5\JQ}{22H}\selectfont\mdgdbfamily}
\def\para@font{\normalsize\bfseries}
\def\subpara@font{\normalsize\bfseries}

\def\section{\@startsection{section}{1}{\z@}% }{
   {1.75\Cvs \@plus.3\Cdp \@minus.1\Cdp}%
   {1.75\Cvs \@plus.3\Cdp \@minus.1\Cdp}%
   {\reset@font\sec@font}}
\def\subsection{\@startsection{subsection}{2}{\z@}% }{
   {1.7\Cvs \@plus.3\Cdp \@minus.1\Cdp}%
   {1.0\Cvs \@plus.3\Cdp \@minus.1\Cdp}%
   {\reset@font\subsec@font}}
\def\subsubsection{\@startsection{subsubsection}{3}{\z@}% }{
   {1\Cvs \@plus.3\Cdp \@minus.1\Cdp}%
   {0\Cvs \@plus.3\Cdp \@minus.1\Cdp}%
   {\reset@font\subsubsec@font}}
\def\paragraph{\@startsection{paragraph}{4}{\z@}% }{
   {\Cvs \@plus.3\Cdp \@minus.1\Cdp}%
   {-1zw}%
   {\reset@font}}
\def\subparagraph{\@startsection{subparagraph}{5}{\z@}% }{
   {\Cvs \@plus.3\Cdp \@minus.1\Cdp}%
   {-1zw}%
   {\reset@font}}

\def\appendix{\par
 \setcounter{chapter}{0}%
 \setcounter{section}{0}%
 \setcounter{figure}{0}%
 \setcounter{table}{0}%
 \setcounter{equation}{0}%
 \renewcommand{\@chapapp}{\appendixname}%
 \renewcommand{\@chappos}{\relax}%
 \renewcommand{\thechapter}{\@Alph\c@chapter}%
}

\if@twocolumn
  \setlength\leftmargini {2zw}% 2em
\else
  \setlength\leftmargini {2zw}% 2.5em
\fi
\setlength\leftmarginii  {2zw}% 2.2em
\setlength\leftmarginiii {2zw}% 1.87em
\setlength\leftmarginiv  {2zw}% 1.7em
\if@twocolumn
  \setlength\leftmarginv {1zw}% .5em
  \setlength\leftmarginvi{1zw}% .5em
\else
  \setlength\leftmarginv {1zw}% 1em
  \setlength\leftmarginvi{1zw}% 1em
\fi
\setlength  \labelsep  {.5zw}% .5em
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty\z@ % -\@lowpenalty
\@endparpenalty\z@ %   -\@lowpenalty
\@itempenalty\z@ %     -\@lowpenalty

\def\@item[#1]{%
  \if@noparitem
    \@donoparitem
  \else
    \if@inlabel
      \indent \par
    \fi
    \ifhmode
      \unskip\unskip \par
    \fi
    \if@newlist
      \if@nobreak
        \@nbitem
      \else
        \addpenalty\@beginparpenalty
        \addvspace\@topsep
        \addvspace{-\parskip}%
      \fi
    \else
      \addpenalty\@itempenalty
      \addvspace\itemsep
    \fi
    \global\@inlabeltrue
  \fi
  \everypar{%
    \@minipagefalse
    \global\@newlistfalse
    \if@inlabel
      \global\@inlabelfalse
      {\setbox\z@\lastbox
       \ifvoid\z@
         \kern-\itemindent
       \fi}%
      \box\@labels
      \penalty\z@
    \fi
    \if@nobreak
      \@nobreakfalse
      \clubpenalty\z@ % \@M --> \z@ 
    \else
      \clubpenalty \@clubpenalty
      \everypar{}%
    \fi}%
  \if@noitemarg
    \@noitemargfalse
    \if@nmbrlist
      \refstepcounter\@listctr
    \fi
  \fi
  \sbox\@tempboxa{\makelabel{#1}}%
  \global\setbox\@labels\hbox{%
    \unhbox\@labels
    \hskip \itemindent
    \hskip -\labelwidth
    \hskip -\labelsep
    \ifdim \wd\@tempboxa >\labelwidth
      \box\@tempboxa
    \else
      \hbox to\labelwidth {\unhbox\@tempboxa}%
    \fi
    \hskip \labelsep}%
  \ignorespaces\inhibitglue
}

\renewcommand{\theenumi}{\@arabic\c@enumi}
\renewcommand{\theenumii}{\@alph\c@enumii}
\renewcommand{\theenumiii}{\@roman\c@enumiii}
\renewcommand{\theenumiv}{\@Alph\c@enumiv}
\newcommand{\labelenumi}{\theenumi.}
\newcommand{\labelenumii}{(\theenumii)}
\newcommand{\labelenumiii}{\theenumiii.}
\newcommand{\labelenumiv}{\theenumiv.}
\renewcommand{\p@enumii}{\theenumi}
\renewcommand{\p@enumiii}{\theenumi(\theenumii)}
\renewcommand{\p@enumiv}{\p@enumiii\theenumiii}

\renewenvironment{enumerate}
  {\ifnum \@enumdepth >3\relax\@toodeep\else
   \advance\@enumdepth\@ne
   \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
   \list{\csname label\@enumctr\endcsname}{%
         \topsep.5\Cvs\parsep\z@\partopsep\z@\itemsep\z@
         \labelwidth2zw
         \labelsep.5zw
         \leftmargin2zw\relax
         \listparindent1zw
         \clubpenalty\z@
         \widowpenalty\z@
         \@itempenalty\z@
         \usecounter{\@enumctr}%
         \def\makelabel##1{\hss\llap{##1}}}%
   \fi}{\endlist}

\newcommand{\labelitemi}{\leavevmode
 \raise.05zh\hbox to 1zw{\hss \textbullet\hss}}
\newcommand{\labelitemii}{\leavevmode\hbox to 1zw{\hss ・\hss}}
\newcommand{\labelitemiii}{\textasteriskcentered}
\newcommand{\labelitemiv}{\textperiodcentered}

\renewenvironment{itemize}
  {\ifnum \@itemdepth >3\relax\@toodeep\else
   \advance\@itemdepth\@ne
   \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
   \expandafter
   \list{\csname \@itemitem\endcsname}{%
         \topsep.5\Cvs\parsep\z@\partopsep\z@\itemsep\z@
         \labelwidth1zw
         \labelsep\z@
         \leftmargin1zw
         \listparindent1zw
         \clubpenalty\z@
         \widowpenalty\z@
         \@itempenalty\z@
         \def\makelabel##1{\hss\llap{##1}}}%
   \fi}{\endlist}

\newenvironment{description}
  {\list{}{\labelwidth\z@ \itemindent-\leftmargin
   \topsep.5\Cvs\parsep\z@\partopsep\z@\itemsep\z@
   \iftdir
     \leftmargin\leftskip \advance\leftmargin3\Cwd
     \rightmargin\rightskip
     \labelsep=1zw \itemsep\z@
     \listparindent\z@
   \fi
           \let\makelabel\descriptionlabel}}{\endlist}
\newcommand{\descriptionlabel}[1]{%
   \hspace\labelsep\normalfont\bfseries #1}

%%shadedab
\newenvironment{shadedab}{%
  \def\FrameCommand{\fcolorbox[gray]{.8}{.8}}%
  \MakeFramed {\advance\hsize-\width \FrameRestore}}%
{\endMakeFramed}
%%end shadedab

%%abstract
\newenvironment{abstract}{
   \begin{shadedab}
    \begin{list}{}{
    \leftmargin=1zw
    \rightmargin=\leftmargin%%%
    }\gt\item[]}{\end{list}
   \end{shadedab}
   \vspace{\baselineskip}}
%%end abstract

%%conclusion
\newenvironment{conclusion}{
   \begin{shadedab}
    \begin{list}{}{
    \leftmargin=1zw
    \rightmargin=\leftmargin%%%
    }\gt\item[]}{\end{list}
   \end{shadedab}
   \vspace{\baselineskip}}
%%end conclusion

%%exbox
\newcommand{\exbox}[1]%
  {\begin{breakbox}
  {#1} 
  \end{breakbox}}
%%end exbox

%%marginpar
\newcounter{marginpar}
\makeatletter
  \let\org@marginpar\marginpar
\renewcommand{\marginpar}[1]{%
  \refstepcounter{marginpar}%  カウンタを一つ増やす
  \kern0pt$^{\themarginpar)}$% 本文での傍注番号
  \org@marginpar{\footnotesize$^{\themarginpar)}$~#1}
  }
\newcommand{\marginparii}[1]{% 番号なしの脚注
  \refstepcounter{marginpar}%  カウンタを一つ増やす
  \org@marginpar{\footnotesize #1}
}

%%end marginpar  
  

\newenvironment{verse}
  {\let\\\@centercr
   \list{}{\itemsep\z@ \itemindent -1.5em%
           \listparindent\itemindent
           \rightmargin\leftmargin \advance\leftmargin 1.5em}%
           \item\relax}{\endlist}

\newenvironment{quotation}
  {\list{}{\listparindent2zw
           \itemindent\listparindent
           \rightmargin\leftmargin
           \topsep.5\Cvs\parsep\z@\partopsep\z@\itemsep\z@
          }%
           \item\relax}{\endlist}

\newenvironment{quote}
  {\list{}{\rightmargin\leftmargin
           \topsep.5\Cvs\parsep\z@\partopsep\z@\itemsep\z@
          }%
           \item\relax}{\endlist}

\newdimen\capwidth

\newcounter{figure}[chapter]
\renewcommand{\thefigure}{%
  \ifnum\c@chapter>\z@\thechapter.\fi\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
%\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure}
\newenvironment{figure}
               {\floatenv\capwidth\hsize
                \abovecaptionskip2mm\belowcaptionskip\z@\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\floatenv\capwidth\hsize
                \abovecaptionskip2mm\belowcaptionskip\z@\@dblfloat{figure}}
               {\end@dblfloat}

\newcounter{table}[chapter]
\renewcommand{\thetable}{\thechapter.\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
%\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\newenvironment{table}
               {\floatenv\capwidth\hsize
                \abovecaptionskip\z@\belowcaptionskip1.5mm\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\floatenv\capwidth\hsize
                \abovecaptionskip\z@\belowcaptionskip1.5mm\@dblfloat{table}}
               {\end@dblfloat}

\newlength\abovecaptionskip
\newlength\belowcaptionskip

%% size in float env.
\def\@floatboxreset{%
 \reset@font
 \small\tabcolsep.5zw
 \@setminipage
}

\def\floatenv{\let\center\Center}% 
\def\Center{\topsep\z@\parsep\z@\partopsep\z@\itemsep\z@
 \trivlist \centering\item\relax}
\def\endCenter{\endtrivlist}

\long\def\@caption#1[#2]#3{\par
 %\addcontentsline{\csname ext@#1\endcsname}{#1}%
 % {\protect\numberline{\csname the#1\endcsname}{\ignorespaces #2}}%
 \begingroup
  \@parboxrestore
  \if@minipage
   \@setminipage
  \fi
   \small
   \@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}\par
 \endgroup}

\newbox\@tempboxb
\let\capnum@font\bfseries

\long\def\@makecaption#1#2{%
 \vskip\abovecaptionskip
 \setbox\@tempboxa\hbox{{\capnum@font #1}\hskip1zw\shortstack[l]{#2}}%
 \@tempdima=\ht\@tempboxa\advance \@tempdima \dp\@tempboxa
 \setbox\@tempboxb\hbox{\capnum@font #1\hskip1zw}%
 \ifdim \@tempdima > \baselineskip
  \ifdim \wd\@tempboxa > \capwidth
   \hfil\parbox[t]{\capwidth}{\hangindent\wd\@tempboxb 
    {\capnum@font #1}\hskip1zw #2}\par
    \hrule height \z@
  \else
   \hfil\parbox[t]{\wd\@tempboxa}{\hangindent\wd\@tempboxb 
    {\capnum@font #1}\hskip1zw #2}\par
    \hrule height \z@
  \fi
 \else
  \ifdim \wd\@tempboxa > \capwidth
   \hfil\parbox[t]{\capwidth}{\hangindent\wd\@tempboxb 
    {\capnum@font #1}\hskip1zw #2}\par
    \hrule height \z@
  \else
   \hbox to\hsize{\hss\box\@tempboxa\hss}%
  \fi
 \fi
 \vspace{\belowcaptionskip}%
}

\arraycolsep5\p@     %% 5\p@
\tabcolsep.5zw       %% 6\p@
\arrayrulewidth.12mm %% .4\p@
\doublerulesep2\p@
\tabbingsep1zw       %% \labelsep

\skip\@mpfootins = \skip\footins
\fboxsep.5zw         %% 3\p@
\fboxrule.12mm       %% .4\p@

\@addtoreset{equation}{chapter}
\renewcommand{\theequation}{%
  \ifnum\c@chapter>\z@\thechapter.\fi \@arabic\c@equation}

\if@enablejfam
  \DeclareSymbolFont{mincho}{JY1}{mc}{m}{n}
  \DeclareSymbolFontAlphabet{\mathmc}{mincho}
  \SetSymbolFont{mincho}{bold}{JY1}{gt}{m}{n}
  \DeclareMathAlphabet\mathgt{JY1}{gt}{m}{n}
  \reDeclareMathAlphabet{\mathrm}{\@mathrm}{\@mathmc}
  \reDeclareMathAlphabet{\mathbf}{\@mathbf}{\@mathgt}
  \jfam\symmincho
\else
  \DeclareRobustCommand{\mathmc}{%
    \@latex@error{Command \noexpand\mathmc invalid with\space
       `disablejfam' class option.}\@eha
  }
  \DeclareRobustCommand{\mathgt}{%
    \@latex@error{Command \noexpand\mathgt invalid with\space
       `disablejfam' class option.}\@eha
  }
\fi

\DeclareOldFontCommand{\mc}{\normalfont\mcfamily}{\mathmc}
\DeclareOldFontCommand{\gt}{\normalfont\gtfamily}{\mathgt}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*{\cal}{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*{\mit}{\@fontswitch\relax\mathnormal}

\setcounter{tocdepth}{2}
\newcommand{\@pnumwidth}{1.55em}
\newcommand{\@tocrmarg}{2.55em}
\newcommand{\@dotsep}{4.5}
\newdimen\toclineskip
\setlength\toclineskip{\z@}
\newdimen\@lnumwidth

\newcommand{\tableofcontents}{%
 \if@twocolumn\@restonecoltrue\onecolumn
  \else\@restonecolfalse\fi
 \chapter*{\contentsname}
 \@mkboth{\contentsname}{\contentsname}%
 \@starttoc{toc}%
 \if@restonecol\twocolumn\fi
}

\newcommand*{\l@part}[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{\baselineskip}%
    \begingroup
    \parindent\z@\rightskip\@pnumwidth
    \parfillskip-\@pnumwidth
    {\leavevmode\large\bfseries
     \setlength\@lnumwidth{4zw}%
    #1\hfil\nobreak%%%%%
     \hbox to\@pnumwidth{\hss#2}}\par
    \nobreak
    \global\@nobreaktrue
    \everypar{\global\@nobreakfalse\everypar{}}%
     \endgroup
  \fi}

\newcommand*{\l@chapter}[2]{%
 \ifnum \c@tocdepth >\m@ne
  \addpenalty{-\@highpenalty}%
  \addvspace{\baselineskip}%

  \vbox to 2\Cvs{\vss%%%%%

  \begingroup
   %\def\numberline##1{\hbox to\@lnumwidth{##1\hfil}}%
   \def\numberline##1{%
    \setbox\@tempboxa\hbox{##1}%
    \@lnumwidth\wd\@tempboxa
    \global\advance \@lnumwidth 1zw
    \hbox to \@lnumwidth{##1\hss}%
   }%
   \parindent\z@ \rightskip\@pnumwidth \parfillskip-\rightskip
   \leavevmode\large\bfseries
    %\setlength\@lnumwidth{4zw}%
    \advance\leftskip\@lnumwidth \hskip-\leftskip
    \inhibitglue
    #1\inhibitglue\nobreak\hfil\nobreak\hbox to\@pnumwidth{\hss #2}\par
    \penalty\@highpenalty
  \endgroup
  \vss}\par
 \fi
}

\newcommand*{\l@section}{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand*{\l@subsection}   {\@dottedtocline{2}{3.8em}{3.2em}}
\newcommand*{\l@subsubsection}{\@dottedtocline{3}{7.0em}{4.1em}}
\newcommand*{\l@paragraph}    {\@dottedtocline{4}{10em}{5em}}
\newcommand*{\l@subparagraph} {\@dottedtocline{5}{12em}{6em}}

%\newcommand{\listoffigures}{%
%  \if@twocolumn\@restonecoltrue\onecolumn
%  \else\@restonecolfalse\fi
%  \chapter*{\listfigurename}
%  \@mkboth{\listfigurename}{\listfigurename}%
%  \@starttoc{lof}%
%  \if@restonecol\twocolumn\fi
%}
%\newcommand*{\l@figure}{\@dottedtocline{1}{1.5em}{2.3em}}
%\newcommand{\listoftables}{%
%  \if@twocolumn\@restonecoltrue\onecolumn
%  \else\@restonecolfalse\fi
%  \chapter*{\listtablename}
%  \@mkboth{\listtablename}{\listtablename}%
%  \@starttoc{lot}%
%  \if@restonecol\twocolumn\fi
%}
%\let\l@table\l@figure

\newdimen\bibindent
\setlength\bibindent{1.5em}
\newcommand{\newblock}{\hskip .11em\@plus.33em\@minus.07em}

\newenvironment{thebibliography}[1]
 {\chapter*{\bibname}%
  \@mkboth{\bibname}{\bibname}%
   \list{\@biblabel{\@arabic\c@enumiv}}%
        {\settowidth\labelwidth{\@biblabel{#1}}%
         \leftmargin\labelwidth
         \advance\leftmargin\labelsep
         \topsep\z@\parsep\z@\partopsep\z@\itemsep\z@
         \@openbib@code
         \usecounter{enumiv}%
         \let\p@enumiv\@empty
         \renewcommand\theenumiv{\@arabic\c@enumiv}}%
   \sloppy
   \clubpenalty\z@
   \widowpenalty\z@
   \sfcode`\.\@m}
  {\def\@noitemerr
    {\@latex@warning{Empty `thebibliography' environment}}%
   \endlist}
\let\@openbib@code\@empty

\def\@biblabel#1{% (
 #1)}

\newenvironment{theindex}
  {\if@twocolumn\@restonecolfalse\else\@restonecoltrue\fi
   \marginparwidth\z@ %追加
   \textwidth154mm %追加
   \oddsidemargin\paperwidth %追加
   \advance \oddsidemargin -\textwidth %追加
   \divide \oddsidemargin \tw@ %追加
   \advance \oddsidemargin -1in %追加
   \evensidemargin\oddsidemargin %追加
   \columnseprule\z@
   \columnsep2zw
   \raggedbottom
   \twocolumn[\@makeschapterhead{\indexname}]%
    %\@mkboth{\indexname}{\indexname}%
   \def\@oddhead{\@Ltop\hfil{\head@font\indexname}\hskip2zw
    {\nombre@font\thepage}\@Rtop}% %追加
   \def\@evenhead{\@Ltop{\nombre@font\thepage}\hskip2zw
    {\head@font\indexname}\hfil\@Rtop}% %追加
    \thispagestyle{empty}%
    \small
    \parindent\z@
    \parskip\z@
    \idx@raggedright
    \setcounter{enumiv}{0}%
    \let\item\@idxitem}
  {\if@restonecol\onecolumn\else\clearpage\fi}

\def\idx@raggedright{%
  \let\\\@centercr
  \@rightskip -.25mm plus 1fil %\@flushglue
  \rightskip\@rightskip
  \leftskip\z@skip
  \parindent\z@}

\newcommand{\@idxitem}{\par\leavevmode\hangindent1zw}
\newcommand{\subitem}{\par\leavevmode\hangindent3zw}
\newcommand{\indexspace}{\vskip\baselineskip}

\newcounter{KANA}
\def\theKANA{\KANAsymbol{KANA}}%
\def\KANAsymbol#1{\@KANAsymbol{\@nameuse{c@#1}}}
\def\@KANAsymbol#1{\ifcase#1\or あ\or か\or さ\or た\or な\or 
 は\or ま\or や\or ら\or わ\else\@ctrerr\fi\relax}

\def\idxhead@font{\small\bfseries}

\def\EGYO{%
 \pagebreak[2]
 \refstepcounter{enumiv}%
 \@tempskipa.5\baselineskip
 \vskip-\prevdepth \prevdepth\z@ \vskip\baselineskip
 \advance\@tempskipa-\baselineskip \vspace*{\@tempskipa}\par
 {\idxhead@font \@Alph\c@enumiv}\par
 \halflineskip
 \nopagebreak
}

\def\GYO{%
 \refstepcounter{KANA}%
 \pagebreak[2]
 \@tempskipa.5\baselineskip
 \vskip-\prevdepth \prevdepth\z@ \vskip\baselineskip
 \advance\@tempskipa-\baselineskip \vspace*{\@tempskipa}\par
 {\idxhead@font \theKANA 行}\par
 \halflineskip
 \nopagebreak
}

\renewcommand{\footnoterule}{%
  \kern-3\p@
  \hrule width .4\columnwidth
  \kern 2.6\p@}

\let\org@footnote\footnote
\def\footnote{\unskip\inhibitglue\penalty\@M\org@footnote}

\@addtoreset{footnote}{chapter}

\def\@makefnmark{\smash{\hbox{\@textsuperscript{\normalfont %(
 \@thefnmark)}}}}

\newcommand\@makefntext[1]{%
 \clubpenalty\@M
 \widowpenalty\@M
 \leftskip1.5zw
 \parindent1zw
 \noindent
 \hbox to \z@{\hss \@makefnmark\,}#1}

\def\eqnarray{%
   \stepcounter{equation}%
   \def\@currentlabel{\p@equation\theequation}%
   \global\@eqnswtrue
   \m@th
   \global\@eqcnt\z@
   \tabskip\@centering
   \let\\\@eqncr
   $$\everycr{}\halign to\displaywidth\bgroup
       \hskip\@centering$\displaystyle\tabskip\z@skip{##}$\@eqnsel
      &\global\@eqcnt\@ne\hskip .8\arraycolsep \hfil${##}$\hfil %% \tw@
      &\global\@eqcnt\tw@ \hskip .8\arraycolsep                 %% 
         $\displaystyle{##}$\hfil\tabskip\@centering
      &\global\@eqcnt\thr@@ \hb@xt@\z@\bgroup\hss##\egroup
         \tabskip\z@skip
      \cr
}
%\def\endeqnarray{%
%      \@@eqncr
%      \egroup
%      \global\advance\c@equation\m@ne
%   $$\@ignoretrue
%}

\def\@begintheorem#1#2{%
 \topsep\z@\parsep\z@\partopsep\z@\itemsep\z@
 \labelsep1zw
 \trivlist
   \item[\hskip \labelsep{\bfseries #1\ #2}]\itshape}
\def\@opargbegintheorem#1#2#3{%
 \topsep\z@\parsep\z@\partopsep\z@\itemsep\z@
 \labelsep1zw
 \trivlist
   \item[\hskip \labelsep{\bfseries #1\ #2\ (#3)}]\itshape}
\def\@endtheorem{\endtrivlist}

\newif\if西暦 \西暦false
\def\西暦{\西暦true}
\def\和暦{\西暦false}
\def\today{{%
  \iftdir
    \if西暦
      \kansuji\number\year 年
      \kansuji\number\month 月
      \kansuji\number\day 日
    \else \@tempcnta\year
      \advance\@tempcnta-1988 平成
      \ifnum \@tempcnta =1 元\else\kansuji\number\@tempcnta\fi 年
      \kansuji\number\month 月\kansuji\number\day 日
    \fi
  \else
    \if西暦
      \number\year\, 年\hspace{.5em}\nobreak
      \number\month\, 月\hspace{.5em}\nobreak
      \number\day\, 日
    \else \@tempcnta\year
      \advance\@tempcnta-1988 平成
      \ifnum \@tempcnta =1 元\else\number\@tempcnta\fi
        \, 年\hspace{.5em}\nobreak
      \number\month\, 月\hspace{.5em}\nobreak\number\day\, 日
    \fi
  \fi}}

\newcommand{\prepartname}{第}
\newcommand{\postpartname}{部}
\newcommand{\prechaptername}{第}
\newcommand{\postchaptername}{章}
\newcommand{\contentsname}{目　次}
\newcommand{\listfigurename}{図目次}
\newcommand{\listtablename}{表目次}
\newcommand{\bibname}{参考文献}
\newcommand{\indexname}{索　引}
\newcommand{\figurename}{図}
\newcommand{\tablename}{表}
\newcommand{\appendixname}{付録}

%%## citesort.sty ##%%
\newcount\@minsofar
\newcount\@min
\newcount\@cite@temp
\def\@citex[#1]#2{%
\if@filesw \immediate \write \@auxout {\string \citation {#2}}\fi
\@tempcntb\m@ne \let\@h@ld\relax \def\@citea{}%
\@min\m@ne%
\@cite{%
  \@for \@citeb:=#2\do {\@ifundefined {b@\@citeb}%
    {\@h@ld\@citea\@tempcntb\m@ne{\bfseries ?}%
    \@warning {Citation `\@citeb ' on page \thepage \space undefined}}%
{\@minsofar\z@ \@for \@scan@cites:=#2\do {%
  \@ifundefined{b@\@scan@cites}%
    {\@cite@temp\m@ne}
    {\@cite@temp\number\csname b@\@scan@cites \endcsname \relax}%
\ifnum\@cite@temp > \@min% select the next one to list
    \ifnum\@minsofar = \z@
      \@minsofar\number\@cite@temp
      \edef\@scan@copy{\@scan@cites}\else
    \ifnum\@cite@temp < \@minsofar
      \@minsofar\number\@cite@temp
      \edef\@scan@copy{\@scan@cites}\fi\fi\fi}\@tempcnta\@min
  \ifnum\@minsofar > \z@ % some more
    \advance\@tempcnta\@ne
    \@min\@minsofar
    \ifnum\@tempcnta=\@minsofar %   Number follows previous--hold on to it
      \ifx\@h@ld\relax
        \edef \@h@ld{\@citea\csname b@\@scan@copy\endcsname}%
      \else \edef\@h@ld{\ifmmode{-}\else--\fi\csname b@\@scan@copy\endcsname}%
      \fi
    \else \@h@ld\@citea\csname b@\@scan@copy\endcsname
          \let\@h@ld\relax
  \fi % no more
\fi}%
\def\@citea{,\penalty\@highpenalty\,}}\@h@ld}{#1}}

%% from "cite.sty", with a little modified
\def\@cite#1#2{\leavevmode\unskip
  \ifnum\lastpenalty=\z@\penalty\@highpenalty\fi% highpenalty before
  \kern\z@
  \smash{%
   $^{\mbox{\normalfont\tiny 
    {\multiply\@highpenalty 3 #1% % triple-highpenalties within
      \if@tempswa,\penalty\@highpenalty\ #2\fi % and before note. %(
    })}}$}\spacefactor\@m}
%% end of cite.sty

\let\org@cite\@cite
\def\cite{\@ifnextchar[% ]
 {\@tempswatrue\let\@cite\org@cite\@citex}
 {\@tempswafalse\let\@cite\org@cite\@citex[]}}
\def\Cite{\@ifnextchar[% ]
 {\@tempswatrue\let\@cite\@Cite\@citex}
 {\@tempswafalse\let\@cite\@Cite\@citex[]}}

\def\@Cite#1#2{\leavevmode %% \unskip
  \ifnum\lastpenalty=\z@\penalty\@highpenalty\fi% highpenalty before
  {\multiply\@highpenalty 3 #1%             % triple-highpenalties within
      \if@tempswa,\penalty\@highpenalty\ #2\fi % and before note.%(
    })\spacefactor\@m}

%% <local defs>
\def\onelineskip{\par\vspace{1\baselineskip}\par}
\def\halflineskip{\par\vspace{.5\baselineskip}\par}
%% </local defs>

\postbreakpenalty`（=10000
\prebreakpenalty`）=10000
\postbreakpenalty`［=10000
\prebreakpenalty`］=10000
\postbreakpenalty`〈=10000
\prebreakpenalty`〉=10000
\prebreakpenalty`"=10000
\prebreakpenalty`℃=10000
\prebreakpenalty`　=10000
\prebreakpenalty`・=10000
\prebreakpenalty`～=10000
\prebreakpenalty`ー=10000
\prebreakpenalty`？=10000
\prebreakpenalty`！=10000
\postbreakpenalty"5C=10000
\postbreakpenalty`【=10000
\prebreakpenalty`】=10000
\postbreakpenalty`〔=10000
\prebreakpenalty`〕=10000
\postbreakpenalty`『=10000
\prebreakpenalty`』=10000

\def\sloppy{\tolerance9999\hbadness9999\hfuzz.5\p@ \vfuzz.5\p@}
%\renewcommand{\fontsubfuzz}{1.1pt}% default is .4pt

\hbadness4000
%\vbadness10001
\tolerance4000\pretolerance\m@ne
\clubpenalty\z@
\widowpenalty\z@

\AtBeginDocument{%
 %% math penalties
 \displaywidowpenalty\z@
 \interdisplaylinepenalty\z@
 \predisplaypenalty\z@
 \postdisplaypenalty\z@
 %%
 %% pstricks
 %\@ifundefined{psset}{}{\psset{unit=1mm}}%
 %% nidanfloat
 \@ifundefined{dblbotfraction}{}{%
  \renewcommand\dblbotfraction{0.9}% 0.5
  \setcounter{dblbotnumber}{4}% 2
 }%
 %% here.sty
 \@ifundefined{@HHfloat}{}{%
  \def\@HHfloat#1[H]{%
   \expandafter\let\csname end#1\endcsname\end@Hfloat
   \vskip\intextsep
   \vbox\bgroup\def\@captype{#1}\parindent\z@
   \@floatboxreset %% added
   \ignorespaces}%
  \def\end@Hfloat{\egroup\vskip-1zh %% added
   \vskip \intextsep}%
 }%
}

\pagestyle{headings}
\pagenumbering{arabic}
\raggedbottom
\if@twocolumn
  \twocolumn
  \sloppy
  \flushbottom
\else
  \onecolumn
\fi
\if@twoside
  \@mparswitchtrue
\else
  \@mparswitchfalse
\fi

\endinput
