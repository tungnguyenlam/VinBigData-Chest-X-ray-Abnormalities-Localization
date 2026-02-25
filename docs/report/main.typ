#import "lib.typ": ieee
#import "@preview/dashy-todo:0.1.3": todo
#import "@preview/abbr:0.3.0"
#import "@preview/mitex:0.2.5": *

#set text(font: "TeX Gyre Termes")
#set figure(placement: auto)

#set heading(
  numbering: "I.A.1.a",
  supplement: none,
)
#show link: underline 

#page[
  #set align(center) 
    #[
      #text(16pt)[UNIVERSITY OF SCIENCE AND TECHNOLOGY OF HANOI]\
      
      #text(13pt)[DEPARTMENT OF INFORMATION AND COMMUNICATION TECHNOLOGY]
      #box(height: 1em)
      #image("usth.jpg", width:60%)\
      #box(height: 1.5em)

      #text(weight: "bold", 40pt)[Computer Vision]\
      #box(height: 5em)
      
      #text(weight: "bold", 39pt)[Final Project Report]\
      #box(height: 2.5em)
  #line(length: 100%)
      #text(weight: "bold",23pt)[
        Chest X-ray Abnormalities Detection
      ]

    #line(length: 100%)
      #text(16pt)[ 
      #text(weight: "bold")[Group] \
      #table(
        stroke: none,
        columns: (auto, auto),
        align: left,
        [Le Chi Thanh Lam], [23BI14248],
        [Nguyen Lam Tung], [],
        [Pham Duy Anh], [23BI14023],
        [Sami GHANNAM], []
      )
      ]
      
    ]
    
    #set align(alignment.bottom)
    #[
      #text(16pt, weight: "bold")[
        Hanoi, 2026
      ]
  ]
]

#set page(footer: context [
  *Chest X-ray Abnormalities Detection*
  #h(1fr)
  #counter(page).display(
    "1/1",
    both: true,
  )
])

#show: ieee.with(
  title: [],
)

#include "abstract.typ"
#pagebreak()
#outline()

#include "introduction.typ"
#include "data.typ"
#include "methodology.typ"
#include "results.typ"
#include "conclusion.typ"

= References
#bibliography("works.bib", full: true)
