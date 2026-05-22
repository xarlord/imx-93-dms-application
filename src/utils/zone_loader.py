"""XML zone polygon loader and classifier."""
from typing import Any

import xml.etree.ElementTree as ET
from .geometry import polygon_area, point_in_polygon


class ZoneLoader:
    """Load zone polygons from an XML file and classify gaze points.

    Parses zone definitions (each with an id, name, type, and polygon) from
    an XML file, then provides methods to classify points and query zones.

    Attributes:
        zones: List of (zone_id, name, type, polygon) tuples.
        zone_order: Zone IDs sorted by ascending polygon area so that
            smaller zones are checked first during classification.
    """

    def __init__(self, xml_path: str) -> None:
        """Initialise the loader and parse the XML zone file.

        Args:
            xml_path: Path to the XML file containing zone definitions.
        """
        self.zones: list[tuple[int, str, str, list[tuple[float, float]]]] = []
        self.zone_order: list[int] = []
        self._load(xml_path)

    def _load(self, xml_path: str) -> None:
        """Parse zone elements from the XML file.

        Args:
            xml_path: Path to the XML file containing zone definitions.

        Raises:
            xml.etree.ElementTree.ParseError: If the XML is malformed.
            FileNotFoundError: If *xml_path* does not exist.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for zone_el in root.findall('zone'):
            zone_id = int(zone_el.get('id'))
            name = zone_el.get('name')
            zone_type = zone_el.get('type', 'off_road')
            poly_el = zone_el.find('polygon')
            points_str = poly_el.get('points')
            polygon: list[tuple[float, float]] = []
            for pair in points_str.split():
                x, y = pair.split(',')
                polygon.append((float(x), float(y)))
            self.zones.append((zone_id, name, zone_type, polygon))

        self.zone_order = [z[0] for z in sorted(self.zones, key=lambda z: polygon_area(z[3]))]

    def classify(
        self, point: tuple[float, float],
    ) -> tuple[int, str, str]:
        """Classify a 2-D point into a zone.

        Zones are tested in ascending area order so that smaller zones
        take priority.

        Args:
            point: The (x, y) point to classify.

        Returns:
            A (zone_id, zone_name, zone_type) tuple. Returns
            (15, 'UNKNOWN', 'off_road') when the point is outside every
            zone.
        """
        for zone_id in self.zone_order:
            for z_id, z_name, z_type, polygon in self.zones:
                if z_id == zone_id and point_in_polygon(point, polygon):
                    return (z_id, z_name, z_type)
        return (15, 'UNKNOWN', 'off_road')

    def is_on_road(self, zone_id: int) -> bool:
        """Check whether a zone is classified as on-road.

        Args:
            zone_id: The zone identifier to look up.

        Returns:
            True if the zone type is ``'on_road'``, False otherwise.
        """
        for z_id, z_name, z_type, _ in self.zones:
            if z_id == zone_id:
                return z_type == 'on_road'
        return False

    def get_zone_by_name(
        self, name: str,
    ) -> tuple[int, str, str, list[tuple[float, float]]] | None:
        """Look up a zone by its name attribute.

        Args:
            name: The zone name string to search for.

        Returns:
            A (zone_id, name, type, polygon) tuple if found, else None.
        """
        for z_id, z_name, z_type, polygon in self.zones:
            if z_name == name:
                return (z_id, z_name, z_type, polygon)
        return None

    def get_all_zones(
        self,
    ) -> list[tuple[int, str, str, list[tuple[float, float]]]]:
        """Return all loaded zone definitions.

        Returns:
            List of (zone_id, name, type, polygon) tuples.
        """
        return self.zones
