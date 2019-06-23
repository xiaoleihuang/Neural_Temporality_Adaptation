#!/bin/bash
for FILE in ./*.pdf; do
  pdfcrop "${FILE}" "${FILE}"
done
