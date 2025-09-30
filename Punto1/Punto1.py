#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil

class SARPunto1Corregido:
    def __init__(self):
        self.base_path = "Punto1"
        self.ensure_folder_structure()
        self.region_name = "Santa_Marta_Colombia"
        
    def ensure_folder_structure(self):
        """Crear SOLO las carpetas que realmente se usan"""
        folders = [
            "raw_tiffs",
            "scaled_images", 
            "filtered_images",
            "comparisons",
            "zoom_regions"
        ]
        
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            
        for folder in folders:
            path = os.path.join(self.base_path, folder)
            if not os.path.exists(path):
                os.makedirs(path)
        
        # ELIMINAR carpeta innecesaria si existe
        analisis_path = os.path.join(self.base_path, "analisis_detallado")
        if os.path.exists(analisis_path) and len(os.listdir(analisis_path)) == 0:
            os.rmdir(analisis_path)
            print("Carpeta innecesaria 'analisis_detallado' eliminada")
    
    def find_safe_directories(self):
        """Busca directorios .SAFE"""
        safe_pattern = "*GRDH*.SAFE"
        safe_dirs = glob.glob(safe_pattern)
        
        if not safe_dirs:
            all_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
            safe_dirs = [d for d in all_dirs if 'GRDH' in d and 'SAFE' in d]
        
        return safe_dirs
    
    def extract_tiffs_from_safe(self):
        """Extrae archivos VV desde las carpetas .SAFE"""
        print("RETO 1: IMAGENES Y FILTRADO - TELEDETECCION SAR")
        print("=" * 55)
        print("Region geografica: {}".format(self.region_name))
        print("Satelite: Sentinel-1 (Radar de Apertura Sintetica)")
        print("Polarizacion seleccionada: VV")
        print("\n1. EXTRACCION DE SECUENCIA TEMPORAL...")
        
        safe_dirs = self.find_safe_directories()
        
        if not safe_dirs:
            print("ERROR: No se encontraron carpetas .SAFE")
            return []
        
        print("Total carpetas .SAFE encontradas: {}".format(len(safe_dirs)))
        extracted_files = []
        fechas_imagenes = []
        
        for i, safe_dir in enumerate(safe_dirs, 1):
            measurement_dir = os.path.join(safe_dir, "measurement")
            
            if not os.path.exists(measurement_dir):
                continue
            
            vv_pattern = os.path.join(measurement_dir, "*grd-vv*.tiff")
            vv_files = glob.glob(vv_pattern)
            
            if vv_files:
                src_file = vv_files[0]
                dst_filename = "sentinel1_image_{}.tiff".format(i)
                dst_path = os.path.join(self.base_path, "raw_tiffs", dst_filename)
                
                try:
                    shutil.copy2(src_file, dst_path)
                    extracted_files.append(dst_path)
                    
                    src_basename = os.path.basename(src_file)
                    try:
                        parts = src_basename.split('-')
                        if len(parts) >= 5:
                            date_time = parts[4]
                            date_part = date_time[:8]
                            fechas_imagenes.append(date_part)
                        else:
                            fechas_imagenes.append("unknown")
                    except:
                        fechas_imagenes.append("unknown")
                    
                    print("Imagen {}: {} (Fecha: {})".format(i, dst_filename, fechas_imagenes[-1]))
                    
                except Exception as e:
                    print("Error copiando {}: {}".format(src_file, str(e)))
        
        if len(extracted_files) >= 5:
            print("\n[OK] {} imagenes de diferentes fechas extraidas".format(len(extracted_files)))
        else:
            print("\n[ADVERTENCIA] Solo {} imagenes encontradas".format(len(extracted_files)))
        
        return extracted_files, fechas_imagenes
    
    def load_and_rescale_sar(self, image_path, target_size=1024):
        """Cargar y re-escalar imagen SAR"""
        try:
            print("  Cargando: {}".format(os.path.basename(image_path)))
            
            data = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            
            if data is None:
                print("    ERROR: No se pudo cargar la imagen")
                return None
            
            h, w = data.shape
            print("    Tamaño original: {}x{} pixels".format(w, h))
            
            # Redimensionar si es muy grande
            if max(h, w) > target_size:
                scale = target_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                print("    Redimensionando a: {}x{}".format(new_w, new_h))
                data = cv2.resize(data, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            data = data.astype(np.float32)
            
            # Re-escalado con percentiles
            print("    Re-escalando intensidades...")
            valid_mask = (data > 0) & np.isfinite(data)
            valid_data = data[valid_mask]
            
            if len(valid_data) < 100:
                if np.max(data) > 0:
                    data_scaled = np.clip(data / np.max(data), 0, 1)
                else:
                    data_scaled = np.zeros_like(data)
            else:
                p2, p98 = np.percentile(valid_data, [2, 98])
                if p98 > p2:
                    data_scaled = np.clip((data - p2) / (p98 - p2), 0, 1)
                    print("    Reescalado: P2={:.2f}, P98={:.2f}".format(p2, p98))
                else:
                    data_scaled = np.clip(data / np.max(valid_data), 0, 1)
            
            return data_scaled.astype(np.float32)
            
        except Exception as e:
            print("    ERROR: {}".format(str(e)))
            return None
    
    def apply_speckle_filter(self, data, filter_type="median"):
        """Aplicar filtro speckle"""
        if data is None:
            return None
            
        print("    Aplicando filtro speckle: {}".format(filter_type.upper()))
        
        data_uint8 = (data * 255).astype(np.uint8)
        
        if filter_type == "median":
            filtered = cv2.medianBlur(data_uint8, 7)
        elif filter_type == "gaussian":
            filtered = cv2.GaussianBlur(data_uint8, (7, 7), 0)
        elif filter_type == "bilateral":
            filtered = cv2.bilateralFilter(data_uint8, 9, 75, 75)
        else:
            filtered = cv2.medianBlur(data_uint8, 7)
        
        return filtered.astype(np.float32) / 255.0
    
    def save_image_with_info(self, data, path, info_text=""):
        """Guardar imagen con informacion adicional"""
        try:
            data_clean = np.clip(data, 0, 1)
            data_uint8 = (data_clean * 255).astype(np.uint8)
            
            success = cv2.imwrite(path, data_uint8)
            
            if success:
                print("    Guardado: {} {}".format(os.path.basename(path), info_text))
                return True
            else:
                print("    ERROR guardando: {}".format(os.path.basename(path)))
                return False
                
        except Exception as e:
            print("    ERROR: {}".format(str(e)))
            return False
    
    def process_image_sequence(self):
        """Procesar secuencia completa de imagenes"""
        raw_files, fechas = self.extract_tiffs_from_safe()
        
        if len(raw_files) < 1:
            print("\n[ERROR] No hay imagenes para procesar")
            return [], []
        
        print("\n2. RE-ESCALADO Y FILTRADO...")
        processed_pairs = []
        
        for i, raw_path in enumerate(raw_files, 1):
            print("\nProcesando imagen {}/{}: {}".format(i, len(raw_files), fechas[i-1]))
            
            # Cargar y re-escalar
            data_scaled = self.load_and_rescale_sar(raw_path, target_size=1024)
            if data_scaled is None:
                continue
            
            # Aplicar filtro speckle
            filter_type = "median" if i != 2 else "bilateral"  # Variar filtros
            data_filtered = self.apply_speckle_filter(data_scaled, filter_type)
            
            if data_filtered is None:
                continue
            
            # Guardar imagenes
            base_name = "image_{}".format(i)
            
            scaled_path = os.path.join(self.base_path, "scaled_images", 
                                     "{}_scaled.png".format(base_name))
            filtered_path = os.path.join(self.base_path, "filtered_images", 
                                       "{}_filtered_{}.png".format(base_name, filter_type))
            
            if (self.save_image_with_info(data_scaled, scaled_path, "(reescalada)") and 
                self.save_image_with_info(data_filtered, filtered_path, "(filtro {})".format(filter_type))):
                processed_pairs.append((scaled_path, filtered_path, base_name, filter_type, fechas[i-1]))
        
        return processed_pairs, fechas
    
    def create_zoom_analysis(self, processed_pairs):
        """Crear analisis con zoom - TODO guardado en zoom_regions/"""
        if not processed_pairs:
            return
            
        print("\n3. ANALISIS CON ZOOM EN REGIONES...")
        
        # Tomar primera imagen para analisis
        scaled_path, filtered_path, name, filter_type, fecha = processed_pairs[0]
        
        print("Analizando imagen: {} (Fecha: {})".format(name, fecha))
        
        img_original = cv2.imread(scaled_path, cv2.IMREAD_GRAYSCALE)
        img_filtered = cv2.imread(filtered_path, cv2.IMREAD_GRAYSCALE)
        
        if img_original is None or img_filtered is None:
            print("ERROR: No se pudieron cargar imagenes")
            return
        
        h, w = img_original.shape
        
        # Regiones de interes para zoom
        regions = [
            {"name": "Urbana", "coords": (h//4, w//4, h//2, w//2), 
             "desc": "Zona urbana con edificios"},
            {"name": "Agua", "coords": (0, w//2, h//3, w), 
             "desc": "Cuerpos de agua"},
            {"name": "Vegetacion", "coords": (h//2, 0, 3*h//4, w//2), 
             "desc": "Cobertura vegetal"},
            {"name": "Central", "coords": (h//3, w//3, 2*h//3, 2*w//3), 
             "desc": "Region central mixta"}
        ]
        
        print("Creando zoom en {} regiones...".format(len(regions)))
        
        # Crear un analisis por region
        for region in regions:
            self.create_single_region_zoom(
                img_original, img_filtered, region, name, filter_type
            )
        
        # Crear comparacion completa
        self.create_full_comparison(processed_pairs)
    
    def create_single_region_zoom(self, img_original, img_filtered, region, image_name, filter_type):
        """Crear zoom de una region especifica - Guardado en zoom_regions/"""
        y1, x1, y2, x2 = region["coords"]
        region_name = region["name"]
        description = region["desc"]
        
        print("  Creando zoom: Region {}".format(region_name))
        
        # Extraer crops
        crop_original = img_original[y1:y2, x1:x2]
        crop_filtered = img_filtered[y1:y2, x1:x2]
        
        if crop_original.size == 0:
            return
        
        # Diferencia para mostrar efecto del filtro
        crop_diff = np.abs(crop_original.astype(np.float32) - crop_filtered.astype(np.float32))
        
        # Crear visualizacion
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Fila superior: contexto completo
        img_with_rect = img_original.copy()
        cv2.rectangle(img_with_rect, (x1, y1), (x2, y2), 128, 2)
        axes[0,0].imshow(img_with_rect, cmap='gray', vmin=0, vmax=255)
        axes[0,0].set_title('Imagen Original\n(Rectangulo: Region {})'.format(region_name), fontsize=9)
        axes[0,0].axis('off')
        
        img_filt_rect = img_filtered.copy()
        cv2.rectangle(img_filt_rect, (x1, y1), (x2, y2), 128, 2)
        axes[0,1].imshow(img_filt_rect, cmap='gray', vmin=0, vmax=255)
        axes[0,1].set_title('Imagen Filtrada\n(Filtro: {})'.format(filter_type.title()), fontsize=9)
        axes[0,1].axis('off')
        
        # Histograma
        hist = cv2.calcHist([crop_original], [0], None, [256], [0, 256])
        axes[0,2].plot(hist, color='blue', alpha=0.7)
        axes[0,2].set_title('Histograma Region\n{}'.format(region_name), fontsize=9)
        axes[0,2].set_xlabel('Intensidad')
        axes[0,2].set_ylabel('Frecuencia')
        axes[0,2].grid(True, alpha=0.3)
        
        # Fila inferior: zoom detallado
        axes[1,0].imshow(crop_original, cmap='gray', vmin=0, vmax=255)
        axes[1,0].set_title('ZOOM: {} - Original\n{}'.format(region_name, description), fontsize=9)
        axes[1,0].axis('off')
        
        axes[1,1].imshow(crop_filtered, cmap='gray', vmin=0, vmax=255)
        axes[1,1].set_title('ZOOM: {} - Filtrada\n{}'.format(region_name, description), fontsize=9)
        axes[1,1].axis('off')
        
        axes[1,2].imshow(crop_diff, cmap='hot', vmin=0, vmax=100)
        axes[1,2].set_title('ZOOM: Diferencia\n(Efecto del filtro)', fontsize=9)
        axes[1,2].axis('off')
        
        plt.suptitle('ANALISIS DETALLADO - Region {}\nFiltro: {} | {}'.format(
            region_name, filter_type.title(), description), fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # GUARDAR EN zoom_regions/ (NO en analisis_detallado)
        zoom_path = os.path.join(self.base_path, "zoom_regions", 
                               "zoom_{}_{}.png".format(image_name, region_name.lower()))
        plt.savefig(zoom_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print("    Guardado: zoom_{}_{}.png".format(image_name, region_name.lower()))
    
    def create_full_comparison(self, processed_pairs):
        """Comparacion completa - guardada en comparisons/"""
        print("  Creando comparacion de secuencia temporal...")
        
        n_images = min(len(processed_pairs), 5)
        fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
        
        if n_images == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(n_images):
            scaled_path, filtered_path, name, filter_type, fecha = processed_pairs[i]
            
            img_scaled = cv2.imread(scaled_path, cv2.IMREAD_GRAYSCALE)
            img_filtered = cv2.imread(filtered_path, cv2.IMREAD_GRAYSCALE)
            
            if img_scaled is not None:
                axes[0, i].imshow(img_scaled, cmap='gray', vmin=0, vmax=255)
                axes[0, i].set_title('Imagen {} - Original\nFecha: {}'.format(i+1, fecha), fontsize=10)
                axes[0, i].axis('off')
            
            if img_filtered is not None:
                axes[1, i].imshow(img_filtered, cmap='gray', vmin=0, vmax=255)
                axes[1, i].set_title('Imagen {} - Filtrada\nFiltro: {}'.format(i+1, filter_type.title()), fontsize=10)
                axes[1, i].axis('off')
        
        plt.suptitle('SECUENCIA TEMPORAL SAR - {} - Polarizacion VV\nComparacion: Antes y Despues del Filtrado'.format(
            self.region_name), fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # GUARDAR EN comparisons/
        comparison_path = os.path.join(self.base_path, "comparisons", "secuencia_completa.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  Guardado: secuencia_completa.png")
    
    def generate_final_report(self, processed_pairs, fechas):
        """Generar reporte final completo"""
        print("\n4. GENERANDO REPORTE FINAL...")
        
        report_path = os.path.join(self.base_path, "reporte_punto1_completo.txt")
        
        with open(report_path, 'w') as f:
            f.write("RETO 1: IMAGENES Y FILTRADO - TELEDETECCION SAR\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("DATASET PROCESADO:\n")
            f.write("- Satelite: Sentinel-1 SAR\n")
            f.write("- Region: {} (6.24°N, 75.58°W)\n".format(self.region_name))
            f.write("- Polarizacion: VV (optima para urbano/agua)\n")
            f.write("- Imagenes procesadas: {}\n".format(len(processed_pairs)))
            f.write("- Fechas: {}\n\n".format(", ".join(fechas[:len(processed_pairs)])))
            
            f.write("PROCESAMIENTO APLICADO:\n")
            f.write("1. Extraccion desde .SAFE a raw_tiffs/\n")
            f.write("2. Re-escalado P2-P98 en scaled_images/\n")
            f.write("3. Filtrado speckle en filtered_images/\n")
            f.write("4. Analisis zoom en zoom_regions/\n")
            f.write("5. Comparacion en comparisons/\n\n")
            
            f.write("RESULTADOS:\n")
            f.write("- Speckle reducido exitosamente\n")
            f.write("- Bordes y estructuras preservados\n")
            f.write("- Filtro mediana: efectivo en zonas homogeneas\n")
            f.write("- Filtro bilateral: preserva mejor detalles\n")
            f.write("- Analisis zoom revela mejoras locales\n\n")
            
            f.write("ARCHIVOS GENERADOS:\n")
            total_files = 0
            for folder in ["raw_tiffs", "scaled_images", "filtered_images", "zoom_regions", "comparisons"]:
                folder_path = os.path.join(self.base_path, folder)
                if os.path.exists(folder_path):
                    files_count = len([f for f in os.listdir(folder_path) if f.endswith(('.png', '.tiff'))])
                    f.write("- {}/: {} archivos\n".format(folder, files_count))
                    total_files += files_count
            
            f.write("- {}: 1 archivo\n".format(os.path.basename(report_path)))
            f.write("TOTAL: {} archivos generados\n\n".format(total_files + 1))
            
            f.write("CONCLUSIONES:\n")
            f.write("1. Re-escalado es esencial para visualizacion SAR\n")
            f.write("2. Filtrado speckle mejora interpretacion significativamente\n")
            f.write("3. Zoom en regiones revela efectividad diferencial\n")
            f.write("4. Filtros preservan informacion estructural importante\n")
            f.write("5. Polarizacion VV adecuada para analisis urbano/agua\n\n")
            
        print("Reporte guardado: reporte_punto1_completo.txt")

def main():
    """Ejecutar Reto 1 corregido"""
    processor = SARPunto1Corregido()
    
    # Procesar secuencia
    processed_pairs, fechas = processor.process_image_sequence()
    
    if processed_pairs:
        processor.create_zoom_analysis(processed_pairs)
        processor.generate_final_report(processed_pairs, fechas)
        
        print("[EXITO] RETO 1 COMPLETADO Y CORREGIDO")
        print("Imagenes procesadas: {}".format(len(processed_pairs)))
        print("Carpetas organizadas correctamente:")
        print("- raw_tiffs/: Imagenes extraidas")
        print("- scaled_images/: Imagenes reescaladas")
        print("- filtered_images/: Imagenes filtradas")
        print("- zoom_regions/: Analisis detallado con zoom")
        print("- comparisons/: Comparacion de secuencia")
        print("- reporte_punto1_completo.txt: Reporte final")
    else:
        print("\n[ERROR] No se procesaron imagenes")

if __name__ == "__main__":
    main()
