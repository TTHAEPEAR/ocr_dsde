"""
Phase 2B: Extract structured fields from raw OCR output.
Handles both constituency vote forms and party-list vote forms.
"""
import re
from loguru import logger


class FieldExtractor:
    """Parse raw OCR text into structured election data fields."""

    # Thai digit mapping (handwritten OCR often produces Thai numerals)
    THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

    def extract(self, raw_results, form_type: str, engine: str) -> dict:
        """
        Main extraction entry point.

        Args:
            raw_results: Raw OCR output (format depends on engine)
            form_type: One of the 6 form types
            engine: "easyocr" or "tesseract"

        Returns:
            dict with extracted fields
        """
        if engine == "easyocr":
            texts, confidences, bboxes = self._parse_easyocr(raw_results)
        elif engine == "tesseract":
            texts, confidences, bboxes = self._parse_tesseract(raw_results)
        else:
            raise ValueError(f"Unknown engine: {engine}")

        # Join all text for full-document parsing
        full_text = " ".join(texts)

        # Extract common fields
        record = {
            "ocr_confidence": sum(confidences) / max(len(confidences), 1),
            "raw_text_preview": full_text[:200],
        }

        # Extract station/constituency identifiers
        record.update(self._extract_identifiers(full_text))

        # Route to form-specific extractor
        if "party" in form_type:
            record.update(self._extract_party_list_votes(texts, bboxes))
        else:
            record.update(self._extract_constituency_votes(texts, bboxes))

        # Extract ballot summary (good/bad/no-vote)
        record.update(self._extract_ballot_summary(full_text))

        return record

    def _parse_easyocr(self, results):
        """Parse EasyOCR output format: [(bbox, text, conf), ...]"""
        texts = [r[1] for r in results]
        confidences = [r[2] for r in results]
        bboxes = [r[0] for r in results]
        return texts, confidences, bboxes

    def _parse_tesseract(self, results):
        """Parse Tesseract output dict format."""
        texts, confidences, bboxes = [], [], []
        for i in range(len(results["text"])):
            text = results["text"][i].strip()
            conf = int(results["conf"][i])
            if text and conf > 0:
                texts.append(text)
                confidences.append(conf / 100.0)
                bboxes.append((
                    results["left"][i], results["top"][i],
                    results["width"][i], results["height"][i]
                ))
        return texts, confidences, bboxes

    def _extract_identifiers(self, text: str) -> dict:
        """Extract station ID, constituency, province from text."""
        result = {}

        # Look for constituency number pattern: "เขตเลือกตั้งที่ X"
        match = re.search(r"เขตเลือกตั้งที่\s*(\d+)", text)
        if match:
            result["constituency_number"] = int(match.group(1))

        # Look for station number: "หน่วยเลือกตั้งที่ X" or just numbers
        match = re.search(r"หน่วย(?:เลือกตั้ง)?ที่\s*(\d+)", text)
        if match:
            result["station_id"] = int(match.group(1))

        # Province name
        match = re.search(r"จังหวัด\s*(\S+)", text)
        if match:
            result["province"] = match.group(1)

        return result

    def _extract_constituency_votes(self, texts: list, bboxes: list) -> dict:
        """
        Extract candidate votes from constituency vote forms (5/16, 5/17, 5/18).
        The form has a table: [number] [party name] [vote count in handwriting]
        """
        votes = {}
        for i, text in enumerate(texts):
            # Normalize Thai digits
            normalized = text.translate(self.THAI_DIGITS)

            # Try to find pattern: party_number followed by vote count
            # The layout is typically: row number | party name | vote count
            numbers = re.findall(r"\d+", normalized)
            if numbers:
                # Heuristic: use spatial position (bbox) to determine
                # if this number is a party index or a vote count
                # Numbers on the right side of the page = vote counts
                if bboxes and len(bboxes) > i:
                    bbox = bboxes[i]
                    x_position = bbox[0][0] if isinstance(bbox[0], list) else bbox[0]

                    # If x > 60% of page width, likely a vote count
                    if x_position > 500:  # Adjust based on your image dimensions
                        vote_count = int(numbers[0])
                        votes[f"candidate_{len(votes)+1}_votes"] = vote_count

        return votes

    def _extract_party_list_votes(self, texts: list, bboxes: list) -> dict:
        """
        Extract party-list votes from party-list forms (5/16บช, 5/17บช, 5/18บช).
        Similar table structure but for party-list ballot.
        """
        votes = {}
        for i, text in enumerate(texts):
            normalized = text.translate(self.THAI_DIGITS)
            numbers = re.findall(r"\d+", normalized)

            if numbers and bboxes and len(bboxes) > i:
                bbox = bboxes[i]
                x_position = bbox[0][0] if isinstance(bbox[0], list) else bbox[0]

                if x_position > 500:
                    vote_count = int(numbers[0])
                    votes[f"party_{len(votes)+1}_votes"] = vote_count

        return votes

    def _extract_ballot_summary(self, text: str) -> dict:
        """Extract ballot summary: good ballots, bad ballots, no-vote ballots."""
        result = {}
        normalized = text.translate(self.THAI_DIGITS)

        # Total ballots: "บัตรดี X บัตร"
        match = re.search(r"บัตรดี\s*(\d+)", normalized)
        if match:
            result["good_ballots"] = int(match.group(1))

        # Bad ballots: "บัตรเสีย X บัตร"
        match = re.search(r"บัตรเสีย\s*(\d+)", normalized)
        if match:
            result["bad_ballots"] = int(match.group(1))

        # No-vote ballots: "ไม่เลือกผู้สมัคร X บัตร"
        match = re.search(r"ไม่เลือก(?:ผู้สมัคร)?\s*(\d+)", normalized)
        if match:
            result["no_vote_ballots"] = int(match.group(1))

        # Total received ballots
        match = re.search(r"(?:รวม|ทั้งหมด|ได้รับบัตร)\s*(\d+)", normalized)
        if match:
            result["total_ballots"] = int(match.group(1))

        return result
