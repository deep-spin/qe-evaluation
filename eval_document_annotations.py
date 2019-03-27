"""Script to evaluate document-level QE as in the WMT19 shared task."""

import argparse
from collections import defaultdict
import numpy as np


class OverlappingSpans(ValueError):
    pass


class Span(object):
    def __init__(self, segment, start, end):
        """A contiguous span of text in a particular segment.
        Args:
            segment: ID of the segment (0-based).
            start: Character-based start position.
            end: The position right after the last character.
        """
        self.segment = segment
        self.start = start
        self.end = end
        assert self.end >= self.start

    def __len__(self):
        """Returns the length of the span in characters.
        Note: by convention, an empty span has length 1."""
        return max(1, self.end - self.start)

    def count_overlap(self, span):
        """Given another span, returns the number of matched characters.
        Zero if the two spans are in different segments.
        Args:
            span: another Span object.
        Returns:
            The number of matched characters.
        """
        if self.segment != span.segment:
            return 0
        start = max(self.start, span.start)
        end = min(self.end, span.end)
        if end >= start:
            if span.start == span.end or self.start == self.end:
                assert start == end
                return 1  # By convention, the overlap with empty spans is 1.
            else:
                return end - start
        else:
            return 0


class Annotation(object):
    def __init__(self, severity=None, spans=None):
        """An annotation, which has a severity level (minor, major, or critical)
        and consists of one or more non-overlapping spans.

        Args:
            severity: 'minor', 'major', or 'critical'.
            spans: A list of Span objects.
        """
        # make sure that there is no overlap
        spans = sorted(spans, key=lambda span: (span.segment, span.start))
        segment = -1
        for span in spans:
            if span.segment != segment:
                # first span in this segment
                segment = span.segment
                last_end = span.end
            else:
                # second or later span
                if span.start < last_end:
                    raise OverlappingSpans()
                last_end = span.end

        self.severity = severity
        self.spans = spans

    def __len__(self):
        """Returns the sum of the span lengths (in characters)."""
        return sum([len(span) for span in self.spans])

    def count_overlap(self, annotation, severity_match=None):
        """Given another annotation with the same severity, returns the number
        of matched characters. If the severities are different, the result is
        penalized according to a severity match matrix.

        Args:
            annotation: another Annotation object.
            severity_match: a dictionary of dictionaries containing match
            penalties for severity pairs.
        Returns:
            The number of matched characters, possibly penalized by a severity
            mismatch.
        """
        # TODO: Maybe normalize by annotation length (e.g. intersection over
        # union)?
        # Note: since we're summing the matches, this won't work as expected
        # if there are overlapping spans (which we assume there aren't).
        matched = 0
        for span in self.spans:
            for annotation_span in annotation.spans:
                matched += span.count_overlap(annotation_span)
        # Scale overlap by a coefficient that takes into account mispredictions
        # of the severity. For example, predicting "major" when the error is
        # "critical" gives some partial credit. If None, give zero credit unless
        # the severity is correct.
        if severity_match:
            matched *= severity_match[self.severity][annotation.severity]
        else:
            matched *= (self.severity == annotation.severity)
        return matched

    @classmethod
    def from_fields(cls, fields):
        """Creates an Annotation object by loading from a list of string fields.

        Args:
            fields: a list of strings containing annotations information. They
                are:
                - segment_id
                - annotation_start
                - annotation_length
                - severity

                The first three fields may contain several integers separated by
                whitespaces, in case there are multiple spans.
                The two last fields are ignored.
                Example: "13 13   229 214 7 4     minor"
        """
        segments = list(map(int, fields[0].split(' ')))
        starts = list(map(int, fields[1].split(' ')))
        lengths = list(map(int, fields[2].split(' ')))
        assert len(segments) == len(starts) == len(lengths)
        severity = fields[3]
        spans = [Span(segment, start, start + length)
                 for segment, start, length in zip(segments, starts, lengths)]
        return cls(severity, spans)

    @classmethod
    def from_string(cls, line):
        """Creates an Annotation object by loading from a string.
        Args:
            line: tab-separated line containing the annotation information. The
                fields are:
                - document_id
                - segment_id
                - annotation_start
                - annotation_length
                - severity

                Segment id, annotation start and length may contain several
                integers separated by whitespaces, in case there are multiple
                spans.
                Example: "A0034 13 13   229 214 7 4     minor"
        """
        # Ignore the last two fields.
        fields = line.split('\t')
        assert len(fields) == 5
        return cls.from_fields(fields[1:])

    def to_string(self):
        """Return a string representation of this annotation.

        This is the representation expected in the output file, without notes"""
        segments = []
        starts = []
        lengths = []
        for span in self.spans:
            segments.append(str(span.segment))
            starts.append(str(span.start))
            lengths.append(str(span.end - span.start))

        segment_string = ' '.join(segments)
        start_string = ' '.join(starts)
        length_string = ' '.join(lengths)
        return '\t'.join([segment_string, start_string, length_string,
                          self.severity])


class Evaluator(object):
    def __init__(self):
        """A document-level QE evaluator."""
        # The severity match matrix will give some credit when the
        # severity is slighted mispredicted ("minor" <> "major" and
        # "major" <> "critical"), but not for extreme mispredictions
        # ("minor" <> "critical").
        self.severity_match = {'minor': {'minor': 1.0,
                                         'major': 0.5,
                                         'critical': 0.0},
                               'major': {'minor': 0.5,
                                         'major': 1.0,
                                         'critical': 0.5},
                               'critical': {'minor': 0.0,
                                            'major': 0.5,
                                            'critical': 1.0}}

    def run(self, system, reference, verbose=False):
        """Given system and reference documents, computes the macro-averaged F1
        across all documents.

        Args:
            system: a dictionary mapping names (doc id's) to lists of
                Annotations produced by a QE system.
            reference: a dictionary mapping names (doc id's) to lists of
                reference Annotations.
        Returns:
            The macro-averaged F1 score.
        """
        total_f1 = 0.
        for doc_id in system:
            # both dicts are defaultdics, returning a empty list if there are no
            # annotations for that doc_id
            reference_annotations = reference[doc_id]
            system_annotations = system[doc_id]
            f1 = self._compare_document(system_annotations,
                                        reference_annotations)
            if verbose:
                print(doc_id)
                print(f1)
            total_f1 += f1
        total_f1 /= len(system)
        return total_f1

    def _compare_document(self, system, reference):
        """Compute the F1 score for a single document, given a system output
        and a reference. This is done by computing a precision according to the
        best possible matching of annotations from the system's perspective,
        and a recall according to the best possible matching of annotations
        from the reference perspective. Gives some partial credit to
        annotations that match with the wrong severity.
        Args:
            system: dictionary mapping doc id's to lists of annotations
            reference: dictionary mapping doc id's to lists of annotations
        Returns:
            The F1 score of a single document.
        """
        all_matched = np.zeros((len(system), len(reference)))
        for i, system_annotation in enumerate(system):
            for j, reference_annotation in enumerate(reference):
                matched = reference_annotation.count_overlap(
                    system_annotation,
                    severity_match=self.severity_match)
                all_matched[i, j] = matched

        lengths_sys = np.array([len(annotation) for annotation in system])
        lengths_ref = np.array([len(annotation) for annotation in reference])

        if lengths_sys.sum() == 0:
            # no system annotations
            precision = 1.
        elif lengths_ref.sum() == 0:
            # there were no references
            precision = 0.
        else:
            # normalize by annotation length
            precision_by_annotation = all_matched.max(1) / lengths_sys
            precision = precision_by_annotation.mean()

        # same as above, for recall now
        if lengths_ref.sum() == 0:
            recall = 1.
        elif lengths_sys.sum() == 0:
            recall = 0.
        else:
            recall_by_annotation = all_matched.max(0) / lengths_ref
            recall = recall_by_annotation.mean()

        if not precision + recall:
            f1 = 0.
        else:
            f1 = 2*precision*recall / (precision + recall)
        assert(0. <= f1 <= 1.)

        return f1


def load_annotations(file_path):
    """Loads a file containing annotations for multiple documents.

    The file should contain lines with the following format:
    <DOCUMENT ID> <LINES> <SPAN START POSITIONS> <SPAN LENGTHS> <SEVERITY>

    Fields are separated by tabs; LINE, SPAN START POSITIONS and SPAN LENGTHS
    can have a list of values separated by white space.

    Args:
        file_path: path to the file.
    Returns:
        a dictionary mapping document id's to a list of annotations.
    """
    annotations = defaultdict(list)

    with open(file_path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            fields = line.split('\t')
            doc_id = fields[0]

            try:
                annotation = Annotation.from_fields(fields[1:])
            except OverlappingSpans:
                msg = 'Overlapping spans when reading line %d of file %s '
                msg %= (i, file_path)
                print(msg)
                continue

            annotations[doc_id].append(annotation)

    return annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('system', help='System annotations')
    parser.add_argument('ref', help='Reference annotations')
    parser.add_argument('-v', help='Show score by document',
                        action='store_true', dest='verbose')
    args = parser.parse_args()

    system = load_annotations(args.system)
    reference = load_annotations(args.ref)
    evaluator = Evaluator()
    f1 = evaluator.run(system, reference, args.verbose)
    print('Final F1:', f1)


if __name__ == '__main__':
    main()
